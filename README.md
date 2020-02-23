# mnexp
Experiments on MSN news recommendation, including LSTUR. For a quick view, please refer to task/paper.py for LSTUR related models.

Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu, Xing Xie. Neural News Recommendation with Long- and Short-term
User Representations. ACL 2019. https://www.aclweb.org/anthology/P19-1033.pdf

# Data preparation
Although the original datasets are not available to the public, I'd like to introduce the data format to visitors.
To run the experiments, the minimum requirement is to prepare ClickData.tsv, DocMeta.tsv and Vocab.tsv. Please refer to settings.py for more details.

In the DocMeta.tsv, there are columns defined as:
`Document Id | Document Index (used in ClickData.tsv to join docs) | Document Category (Vertical) | Document Sub-category (Subvertical) | Document Title | Document Content (Body) | ...(columns not necessary for this project)`
For all these tsv files, each columns are separated by TAB. Though not necessary, we mapped news title and news body to the vocabulary index. Besides, you can find the news verticals and subverticals in utils.py.

The Vocab.tsv stores words and embeddings in the format 
`word index | word word embedding (separated by space for each dimension)`
We highly recommend to use public word embeddings, or embeddings trained from news articals covering more than 10 years, other than which trained from those articals in a short period. 

We collect user click history in the ClickData.tsv, where columns are defined as:
`User Id (not used) | User Type (not used) | User Click History in the period for training | User Click History in the period for validation | ...(columns not necessary for this project)`
For the user click history, we arrange the history by impressions (page visit), e.g. 
`36 27#TAB#18 49 51 77 62 83 5#TAB#01/18/2019 08:15:31 am#TAB#(fields not necessary for this project)#N#96#TAB#62 43 59 81 77 143 12#TAB#01/18/2019 08:54:18 am#TAB#(fields not necessary for this project)`
which shows that the user has two page visits on 01/18/2019 08:15:31 am and 01/18/2019 08:54:18 am (separated by #N#). For the first impression, the service showed the user several news candidates, and the user clicked news #36 and #27 for details and ignored #18, #49, #51, #77, #62, #83, #5; for the second one, #96 was clicked and #62, #43, #59, #81, #77, #143, #12 were ignored.
