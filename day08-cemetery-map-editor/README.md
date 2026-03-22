# Day 08: Cemetery Map Editor Prototype

This day's project captures an interactive prototype for editing a cemetery map layout. The HTML page mirrors the hosted draft from [cemetery_page_draft_5.html](https://alexander-joseph-hamburger.github.io/Benjamin/cemetery_page_draft_5.html) so it can be version controlled within the challenge repository.

## Files

- `cemetery_page_draft_5.html` – Standalone HTML file that includes styles, layout logic, and JavaScript helpers required to render and manipulate the cemetery grid editor in the browser. The editor can fetch cemetery map data from a web-hosted JSON file.
- `sample-cemetery-map.json` – Minimal example dataset used by default when the page is served over HTTP/HTTPS.

## Usage

Serve the HTML file via a simple HTTP server (or host it on GitHub Pages) and open it in a modern browser. The “Map JSON URL” field defaults to `sample-cemetery-map.json`, enabling the editor to fetch data from the hosted JSON file. Replace the URL with any other publicly reachable JSON endpoint that matches the expected schema to load different cemetery layouts.
