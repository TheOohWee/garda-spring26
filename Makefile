.PHONY: install test lint run-all scrape clean build-csv score report

install:
	pip install -e .

test:
	python3 -m unittest discover -s tests -v

run-all:
	python3 -m garda1.cli run-all --seed-archive --write-speakers

scrape:
	python3 -m garda1.cli scrape --seed-archive

clean:
	python3 -m garda1.cli clean --write-speakers

build-csv:
	python3 -m garda1.cli build-csv

score:
	python3 -m garda1.cli score

report:
	python3 -m garda1.cli report
