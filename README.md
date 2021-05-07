## Final Project -- Udacity Introduction to Machine Learning (ud-120)

This project was my introduction to coding ML algorithms using [scikit-learn](https://scikit-learn.org).
The course at the time I took it was no longer being evaluated/graded, so I may not have followed every single step in the rubric.
I left this in Jupyter form for other newbies who may want to see the full process of EDA & developing an algorithm in all its glory.

### Background
The goal of the project was to use financial and email data related to the infamous [Enron](https://en.wikipedia.org/wiki/Enron) financial scandal in an ML model to identify Persons of Interest (POIs) in the subsequent investigation & criminal prosecutions.

There definition of POI was anyone who was:
-- indicted
-- settled without admitting guilt
-- testified in exchange for immunity  

The list of POIs they obtained based on these criteria are in `poi_names.txt`, along with a link to the background article.

The original financial data from the discovery process are in `enron61702insiderpay.pdf`.

The full email corpus is available [here](https://www.cs.cmu.edu/~enron/).  
It's not necessary for this exercise, but might be interesting for a more advanced text classification project.

The data necessary for the project have already been aggregated into the file `final_project_dataset.pkl`.
This includes:
* financial information (see the pdf above for details; I also ended up looking up many of terms on [Investopedia](https://www.investopedia.com/financial-term-dictionary-4769738))
* POI status (in the field 'poi')
* Email meta-data, such as:
  - # messages from this person (`from_messages`)
  - # messages to this person (`to_messages`)
  - # messages from this person to POIs (`from_this_person_to_poi`)
  - # messages from POIs to this person (`from_poi_to_this_person`)
  - # messages where person & POI were both recipients (`shared_receipt_with_poi`)

The dataset is small. There are 146 employees, and many of the fields are missing data.
The goal was to obtain an ML model that could predict POI status with precision & recall > 0.3.

### Usage
The notebook code is in Python 3 format.  
The dataset `final_project_dataset.pkl` should be in the same directory (or change the path).  

You will also need:
-- `scikit-learn`
-- `numpy`
-- `pandas`
-- `matplotlib`
-- `seaborn`

Beyond that, just work your way through the notebook.
See if you can identify better features or models, or find a way to keep the dataset from shrinking the way it did on me.  

The intrinsic features I used were:
- `salary`
- `total_payments`
- `bonus`

And the derived I created were:
- `stock+lti` (sum of total stock value and long-term incentives)
- `pct_to_poi` (% of a person's emails that are addresed to a POI)
- `pct_shared_receipt` (% of a person's emails for which they're on a recipient list with a POI)

I tried Naive Bayes and Support Vector Machine models, as well as PCA for dimensionality reduction. GridSearchCV & manual tuning were used to optimize the hyperparameters of the SVM.  
The best result I obtained (using a train/test split of 0.67/0.33) was for an SVM without the PCA-transformed features.  The optimal parameters, obtained from manual fine-tuning, were `kernel` = RBF, `C` = 93, and `gamma` = 0.18. The accuracy of this model was 0.73, and precision, recall, and f1 were all 0.5.   
Not stunning, but they met the project goal. The small dataset was a big limitation.

## History
March 2, 2021
Updated May 7, 2021

## License  
[Licensed](license.md) under the [MIT License](https://spdx.org/licenses/MIT.html).
