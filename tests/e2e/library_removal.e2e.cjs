// Browser test of "Remove from this library" against a running, DISPOSABLE library.
//
//   NODE_PATH=<dir with playwright-core>/node_modules node tests/e2e/library_removal.e2e.cjs \
//     --base http://127.0.0.1:8002 --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe" \
//     --cancel OM2NF.mp4 --remove GG4UR.mp4 --gone I2DHG.mp4 [--in-use OM2NF.mp4] --shots <dir>
//
// --cancel   open the dialog, cancel (button and Escape): no request, nothing changes
// --remove   confirm: success toast, row and filter option disappear at once
// --gone     open the dialog, remove the video out of band, confirm: 404 error toast
// --in-use   a file of this video is held open by another process: 409 error toast, row stays
// Never point it at a library you want to keep: it really removes videos (recoverably).
const { chromium } = require("playwright-core");
const assert = require("node:assert/strict");
const path = require("node:path");

const args = {};
for (let i = 2; i < process.argv.length; i += 2) args[process.argv[i].replace(/^--/, "")] = process.argv[i + 1];
const base = args.base;
const shot = (page, name) => args.shots && page.screenshot({ path: path.join(args.shots, `${name}.png`) });

async function listVideos() {
  return (await (await fetch(`${base}/api/videos`)).json()).map((video) => video.display_name).sort();
}

async function main() {
  const browser = await chromium.launch({ executablePath: args.chrome, headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const deletes = [];
  page.on("request", (request) => { if (request.method() === "DELETE") deletes.push(request.url()); });
  const row = (name) => page.locator(".video-row", { has: page.locator("h3", { hasText: name }) });
  const results = [];
  const pass = (name) => { results.push(name); console.log(`PASS ${name}`); };

  await page.goto(`${base}/#library`);
  const initial = await listVideos();
  await page.waitForFunction((count) => document.querySelectorAll(".video-row").length === count, initial.length);

  if (args.cancel) {
    await row(args.cancel).locator(".remove-video").click();
    await page.locator("#remove-dialog[open]").waitFor();
    assert.equal(await page.locator("#remove-name").textContent(), args.cancel);
    await shot(page, "1-confirm-dialog");
    await page.locator("#remove-cancel").click();
    await page.locator("#remove-dialog[open]").waitFor({ state: "detached" });
    await row(args.cancel).locator(".remove-video").click();
    await page.locator("#remove-dialog[open]").waitFor();
    await page.keyboard.press("Escape");
    await page.locator("#remove-dialog[open]").waitFor({ state: "detached" });
    await page.waitForTimeout(800);
    assert.deepEqual(deletes, []);
    assert.deepEqual(await listVideos(), initial);
    assert.equal(await page.locator(".video-row").count(), initial.length);
    pass(`cancel and Escape send no request and change nothing (${args.cancel})`);
  }

  if (args.remove) {
    await row(args.remove).locator(".remove-video").click();
    await page.locator("#remove-dialog[open]").waitFor();
    assert.equal(await page.locator("#remove-name").textContent(), args.remove);
    await page.locator("#remove-confirm").click();
    const toast = page.locator("#toast.success");
    await toast.waitFor();
    assert.match(await toast.textContent(), new RegExp(`Removed “${args.remove.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}”`));
    assert.equal(await row(args.remove).count(), 0);
    const options = await page.locator("#video-filter option").allTextContents();
    assert.ok(!options.includes(args.remove), "filter still offers the removed video");
    assert.ok(!(await listVideos()).includes(args.remove));
    assert.equal(deletes.length, 1);
    await shot(page, "2-success-toast");
    pass(`confirmed removal shows success, updates list and filter at once (${args.remove})`);
  }

  if (args.gone) {
    await row(args.gone).locator(".remove-video").click();
    await page.locator("#remove-dialog[open]").waitFor();
    const id = (await (await fetch(`${base}/api/videos`)).json()).find((video) => video.display_name === args.gone).video_id;
    assert.equal((await fetch(`${base}/api/videos/${id}`, { method: "DELETE" })).status, 200);
    await page.locator("#remove-confirm").click();
    const toast = page.locator("#toast.error");
    await toast.waitFor();
    assert.match(await toast.textContent(), /Video was not found in this library/);
    await page.waitForFunction((name) => ![...document.querySelectorAll(".video-row h3")].some((h) => h.textContent === name), args.gone);
    await shot(page, "3-not-found-toast");
    pass(`removing a video already removed elsewhere shows the server error (${args.gone})`);
  }

  if (args["in-use"]) {
    const before = await listVideos();
    await page.locator("#toast").evaluate((toast) => toast.classList.add("hidden"));
    await row(args["in-use"]).locator(".remove-video").click();
    await page.locator("#remove-dialog[open]").waitFor();
    await page.locator("#remove-confirm").click();
    const toast = page.locator("#toast.error");
    await toast.waitFor();
    assert.match(await toast.textContent(), /in use/);
    await page.waitForFunction((name) => {
      const card = [...document.querySelectorAll(".video-row")].find((item) => item.querySelector("h3").textContent === name);
      return card && !card.classList.contains("removing") && !card.querySelector(".remove-video").disabled;
    }, args["in-use"]);
    assert.deepEqual(await listVideos(), before);
    await shot(page, "4-in-use-toast");
    pass(`a file in use returns 409, the row stays and the library is unchanged (${args["in-use"]})`);
  }

  await browser.close();
  console.log(`${results.length} browser checks passed`);
}

main().catch((error) => { console.error(error); process.exit(1); });
