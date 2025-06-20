
all: divselement.pdf

divselement.pdf: divselement.tex
	xelatex -shell-escape divselement.tex
	#bibtex divselement.aux
	xelatex -shell-escape divselement.tex
	xelatex -shell-escape divselement.tex
	evince divselement.pdf&

.PHONY:clean  
clean:
	-rm -fr *.xml *.bcf *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *~ *.tid *.tms *.pdf *.bak *.blg *.bbl *.gz *.snm *.nav _minted*

