# CRSArena-Eval Interactive Interface

An interactive web-based evaluation tool for CRS Arena benchmarks. Upload your run file and instantly see correlation metrics (Pearson & Spearman) for turn-level and dialogue-level aspects.

**Run locally:**

Navigate to the `interface` directory and start a simple HTTP server (CORS restrictions prevent direct file opening). Then open `http://localhost:8000` in your browser.

```bash
cd interface
python3 -m http.server 8000
```

The public interface is hosted at
[`https://informagi.github.io/face/interface/`](https://informagi.github.io/face/interface/).

For run file format, see: [`dataset/run/README.md`](../dataset/run/README.md)

## Demo

![CRSArena-Eval demo](../demo/demo.gif)
