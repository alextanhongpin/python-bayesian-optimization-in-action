# cp ../Makefile Makefile
#import sys

#!"{sys.executable}" --version
#!which "{sys.executable}"
#!jupyter labextension list

include .env
export

jupyter: 
	@poetry run jupyter-lab

kernel:
	poetry run python -m pip install ipykernel
	poetry run python -m ipykernel install --user

install:
	poetry add jupyterlab-vim jupyterlab-code-formatter ipywidgets black isort

convert_all:
	# jupytext doesn't preserve image.
	#@find . -name "*.ipynb" ! -path '*/.*' -exec poetry run jupytext --to md {} \;
	@find . -name "*.ipynb" ! -path '*/.*' -exec poetry run jupyter nbconvert --to markdown --output-dir=docs {} \;

# Similar to convert, but only convert the diff files.
# You need to 'git add .' the files before running this command.
convert:
	poetry run jupyter nbconvert --to markdown --output-dir=docs $(shell git diff HEAD --name-only | grep .ipynb)


lint:
	@poetry run black *.py

