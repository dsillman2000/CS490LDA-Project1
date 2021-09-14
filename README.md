# CS490LDA-Project1
Project worked on by David Sillman, Joshua Lefton, Devon Herr, and Parinith Rajkumar for CS 490 LDA at Purdue University, Fall 2021.

## Reference Paper
View [here](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2018_2019/papers/Kraska_SIGMOD_2018.pdf).

## Organization

Models are included in the `\models` folder.
Testing files are in the `\testing` folder, to which `\testing\data` contains files for benchmarking.
Generally these are `.csv` files where rows represent a given dataset size, and columns represent the number of queries, each cell being the average number of time for the combination.

### Running benchmark scripts

As we call/import python functions from other files that are in different folders, the standard `python testing\script.py` command line call will not work.
To that end, using the standard run functionality in most IDEs will not suffice.

Instead, use
```
python -m testing.script
```
by replacing `\`s with `.`s, and removing the `.py` extension.
You can think of this as using the syntax for imports---using the `-m` treats this as a module.

