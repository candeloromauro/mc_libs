# mc_libs LaTeX Documentation

This folder contains an auto-generated comprehensive guide for all `mc_*` modules and public APIs.

## Files

- `generate_library_guide.py`: scans the repository and generates the LaTeX source.
- `mc_libs_guide.tex`: generated LaTeX guide.
- `mc_libs_guide.pdf`: compiled PDF guide.

## Regenerate the guide

```sh
python3 /Users/mcandeloro/python_projects/mc_libs/docs/generate_library_guide.py
```

## Compile the PDF

```sh
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory /Users/mcandeloro/python_projects/mc_libs/docs \
  /Users/mcandeloro/python_projects/mc_libs/docs/mc_libs_guide.tex
```

Run `pdflatex` twice if table of contents links need refresh.
