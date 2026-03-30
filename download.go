package main

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
)

// termWriter manages cursor positioning so each file owns a fixed row
type termWriter struct {
	mu      sync.Mutex
	rowCount int
}

var term = &termWriter{}

// reserve claims the next available row, returning its index
func (tw *termWriter) reserve() int {
	tw.mu.Lock()
	defer tw.mu.Unlock()
	row := tw.rowCount
	tw.rowCount++
	fmt.Println() // push the terminal down to make room for this row
	return row
}

// write moves to the reserved row, renders, then returns the cursor to the bottom
func (tw *termWriter) write(row int, line string) {
	tw.mu.Lock()
	defer tw.mu.Unlock()
	linesUp := tw.rowCount - row
	// Move up, rewrite the line, move back to bottom
	fmt.Printf("\033[%dA\r%s\033[%dB\r", linesUp, line, linesUp)
}

// progressReader wraps an io.Reader, tracking bytes read via an atomic counter
type progressReader struct {
	reader    io.Reader
	bytesRead *atomic.Int64
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.reader.Read(p)
	pr.bytesRead.Add(int64(n))
	return n, err
}

func renderBar(filename string, current, total int64) string {
	const barWidth = 30
	var pct float64
	if total > 0 {
		pct = float64(current) / float64(total)
	}
	filled := int(pct * float64(barWidth))
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)
	return fmt.Sprintf("%-20s [%s] %5.1f%%  %s/%s",
		truncate(filename, 20),
		bar,
		pct*100,
		formatBytes(current),
		formatBytes(total),
	)
}

// printProgress owns a reserved terminal row and updates it until done is closed
func printProgress(row int, filename string, bytesRead *atomic.Int64, total int64, done <-chan struct{}) {
	for {
		select {
		case <-done:
			term.write(row, renderBar(filename, total, total))
			return
		default:
			term.write(row, renderBar(filename, bytesRead.Load(), total))
		}
	}
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-3] + "..."
}

func formatBytes(b int64) string {
	switch {
	case b >= 1 << 30:
		return fmt.Sprintf("%.1fGB", float64(b)/float64(1<<30))
	case b >= 1 << 20:
		return fmt.Sprintf("%.1fMB", float64(b)/float64(1<<20))
	case b >= 1 << 10:
		return fmt.Sprintf("%.1fKB", float64(b)/float64(1<<10))
	default:
		return fmt.Sprintf("%dB", b)
	}
}

func workerCalc(fileSize int64) int64 {
	workers := fileSize / (500 * 1024 * 1024)
	if workers < 1 {
		workers = 1
	}
	if workers > 8 {
		workers = 8
	}
	return workers
}

func readParse() ([]string, error) {
	file, err := os.Open("ava_train_v2.2.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to open file list: %w", err)
	}
	defer file.Close()

	var files []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		if line := scanner.Text(); line != "" {
			files = append(files, line)
		}
	}
	return files, scanner.Err()
}

func Download(baseURL, filename string) error {
	fullURL := baseURL + filename

	resp, err := http.Head(fullURL)
	if err != nil {
		return fmt.Errorf("HEAD request failed for %s: %w", filename, err)
	}
	resp.Body.Close()

	fileSize := resp.ContentLength
	numWorkers := workerCalc(fileSize)
	chunkSize := fileSize / numWorkers

	outFile, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file %s: %w", filename, err)
	}
	defer outFile.Close()

	var bytesRead atomic.Int64
	done := make(chan struct{})

	// Reserve a row before spawning workers so the row exists before any writes
	row := term.reserve()
	go printProgress(row, filename, &bytesRead, fileSize, done)

	var wg sync.WaitGroup
	errs := make(chan error, numWorkers)

	var i int64
	for i = 0; i < numWorkers; i++ {
		wg.Add(1)
		start := i * chunkSize
		end := start + chunkSize - 1
		if i == numWorkers-1 {
			end = fileSize - 1
		}

		go func(start, end int64) {
			defer wg.Done()
			if err := downloadChunk(fullURL, outFile, start, end, &bytesRead); err != nil {
				errs <- err
			}
		}(start, end)
	}

	wg.Wait()
	close(done)
	close(errs)

	for err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

func downloadChunk(url string, out *os.File, start, end int64, bytesRead *atomic.Int64) error {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("chunk request failed: %w", err)
	}
	defer resp.Body.Close()

	pr := &progressReader{reader: resp.Body, bytesRead: bytesRead}
	data, err := io.ReadAll(pr)
	if err != nil {
		return fmt.Errorf("failed to read chunk body: %w", err)
	}

	_, err = out.WriteAt(data, start)
	return err
}

func main() {
	const baseURL = "https://s3.amazonaws.com/ava-dataset/trainval/"
	const maxConcurrentFiles = 5

	listOfFiles, err := readParse()
	if err != nil {
		panic(err)
	}

	sem := make(chan struct{}, maxConcurrentFiles)
	var wg sync.WaitGroup

	for _, filename := range listOfFiles {
		wg.Add(1)
		sem <- struct{}{}
		go func(f string) {
			defer wg.Done()
			defer func() { <-sem }()
			if err := Download(baseURL, f); err != nil {
				fmt.Fprintf(os.Stderr, "error downloading %s: %v\n", f, err)
			}
		}(filename)
	}

	wg.Wait()
	fmt.Println("\nAll downloads complete.")
}