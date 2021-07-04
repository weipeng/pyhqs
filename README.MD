# Python Code for Hierarchy Quality of Search

The Amazon qulitative analysis is ad-hoc. 
Thus, the process steps are a bit annoying. Please follow the storyline.

## Sampling the Amazon Items    
The items are sampled from the file `metadata.json`, which refers to the Amazon dataset. 
If that is not in your computer, try to map the items using the AI journal draft (in another repo).


## Setting up the tree
The trees are hand made.
And the notebook for generating the trees and plots is `./notebook/Draw trees.ipynb`.
Follow the procedure in the notebook, we got to save two important files:
`amazon_tree_{}_Z.json` and `amazon_tree_{}.pdf` where the index should be put into the bracket.
Copy the json file to the folder `./data` and copy the pdf file to the folder of the journal.

## Running HQS
For analyzing the 12 Amazon items, we first uncommnet all the logging code in the file `./polices/greedypolicy.py`.
Then, use the following command:
```
python -m simulations.analyze amazon index
```
The index matches the index of the tree you would like to analyze.
After that is done, type in 
```
mv ./logs/log.txt ./logs/amazon{}.log
```
where the index should be put into the bracket.

For other experiments, checkout the file `./simulations/analyze.py` for more options.


## HQS output
The output is scored in `./output/scores/amazon_{}.csv` where the index matches the index you used for running the analysis.
The file `./notebook/Amazon tree to latex.ipynb` can be used to generate the latex for the HQS table.
However, for plugging the table into the journal draft, it needs some more minor latex coding, e.g. adding `\begin{table}[ht]` etc.






# Setting delta in the guidance function
Unfortunately, there are some hard-setting parameters that the users need to change within the code.
One has to set `delta`, which is associated to the guidance function, in the script `./policies/greedypolicy.py`.
