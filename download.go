package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// termWriter manages cursor positioning so each file owns a fixed row
type termWriter struct {
	mu       sync.Mutex
	rowCount int
}

var term = &termWriter{}
var httpClient = &http.Client{Timeout: 5 * time.Minute}

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
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			term.write(row, renderBar(filename, total, total))
			return
		case <-ticker.C:
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
	case b >= 1<<30:
		return fmt.Sprintf("%.1fGB", float64(b)/float64(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1fMB", float64(b)/float64(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.1fKB", float64(b)/float64(1<<10))
	default:
		return fmt.Sprintf("%dB", b)
	}
}

func workerCalc(fileSize int64) int64 {
	workers := fileSize / (64 * 1024 * 1024)
	if workers < 1 {
		workers = 1
	}
	if workers > 4 {
		workers = 4
	}
	return workers
}

func readParse(listPath string) ([]string, error) {
	file, err := os.Open(listPath)
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

func Download(baseURL, filename, outputDir string) error {
	fullURL := baseURL + filename
	destination := filepath.Join(outputDir, filename)

	req, err := http.NewRequest(http.MethodHead, fullURL, nil)
	if err != nil {
		return fmt.Errorf("HEAD request creation failed for %s: %w", filename, err)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("HEAD request failed for %s: %w", filename, err)
	}
	resp.Body.Close()

	fileSize := resp.ContentLength
	if resp.StatusCode < 200 || resp.StatusCode >= 300 || fileSize <= 0 {
		return fmt.Errorf("HEAD request for %s returned %s", filename, resp.Status)
	}
	if info, statErr := os.Stat(destination); statErr == nil && info.Size() == fileSize {
		fmt.Printf("%-20s cached (%s)\n", truncate(filename, 20), formatBytes(fileSize))
		return nil
	}
	numWorkers := workerCalc(fileSize)
	chunkSize := fileSize / numWorkers

	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	partial := destination + ".part"
	outFile, err := os.Create(partial)
	if err != nil {
		return fmt.Errorf("failed to create output file %s: %w", partial, err)
	}
	if err := outFile.Truncate(fileSize); err != nil {
		outFile.Close()
		return fmt.Errorf("failed to size partial file %s: %w", partial, err)
	}

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
			if err := downloadChunkWithRetry(fullURL, outFile, start, end, &bytesRead); err != nil {
				errs <- err
			}
		}(start, end)
	}

	wg.Wait()
	close(done)
	close(errs)

	for err := range errs {
		if err != nil {
			outFile.Close()
			return err
		}
	}
	if err := outFile.Sync(); err != nil {
		outFile.Close()
		return err
	}
	if err := outFile.Close(); err != nil {
		return err
	}
	if err := os.Rename(partial, destination); err != nil {
		return fmt.Errorf("failed to finalize %s: %w", destination, err)
	}
	return nil
}

func downloadChunk(url string, out *os.File, start, end int64, bytesRead *atomic.Int64) error {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", start, end))

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("chunk request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("range %d-%d returned %s", start, end, resp.Status)
	}

	pr := &progressReader{reader: resp.Body, bytesRead: bytesRead}
	written, err := io.Copy(io.NewOffsetWriter(out, start), pr)
	if err != nil {
		return fmt.Errorf("failed to stream chunk body: %w", err)
	}
	expected := end - start + 1
	if written != expected {
		return fmt.Errorf("range %d-%d wrote %d bytes, expected %d", start, end, written, expected)
	}
	return nil
}

func downloadChunkWithRetry(url string, out *os.File, start, end int64, bytesRead *atomic.Int64) error {
	var lastErr error
	for attempt := 1; attempt <= 4; attempt++ {
		lastErr = downloadChunk(url, out, start, end, bytesRead)
		if lastErr == nil {
			return nil
		}
		if attempt < 4 {
			time.Sleep(time.Duration(attempt*attempt) * time.Second)
		}
	}
	return fmt.Errorf("range %d-%d failed after 4 attempts: %w", start, end, lastErr)
}

func main() {
	listPath := flag.String("list", "ava_train_v2.2.txt", "newline-delimited AVA filenames")
	outputDir := flag.String("output-dir", ".", "download destination")
	baseURL := flag.String("base-url", "https://s3.amazonaws.com/ava-dataset/trainval/", "AVA media base URL")
	flag.Parse()
	const maxConcurrentFiles = 2

	listOfFiles, err := readParse(*listPath)
	if err != nil {
		panic(err)
	}

	sem := make(chan struct{}, maxConcurrentFiles)
	var wg sync.WaitGroup
	var failed atomic.Bool

	for _, filename := range listOfFiles {
		wg.Add(1)
		sem <- struct{}{}
		go func(f string) {
			defer wg.Done()
			defer func() { <-sem }()
			if err := Download(*baseURL, f, *outputDir); err != nil {
				failed.Store(true)
				fmt.Fprintf(os.Stderr, "error downloading %s: %v\n", f, err)
			}
		}(filename)
	}

	wg.Wait()
	if failed.Load() {
		fmt.Fprintln(os.Stderr, "one or more downloads failed")
		os.Exit(1)
	}
	fmt.Println("\nAll downloads complete.")
}
