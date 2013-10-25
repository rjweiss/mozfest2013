# Document-level text analysis

Document-level analysis is when you are interested in the whole text article, not tokens (sentences or words).  The most basic example is labeling documents against some classification scheme, hence **text classification**.  When you don't know your scheme ahead of time or you're interested in exploring a large set of data, you can try **topic modeling**.

We're going to go over a couple of examples of document-level text analysis using some very most common classifiers models.  We're going to go over the code to train your own model and discuss the results we see.

## Supervised learning: Text classification in R

We're going to go over examples of how to use the excellent [RTextTools](http://www.rtexttools.com/) library to train some text classifiers.

The dataset used are the titles and topic codes from the `NYTimes` dataset that comes with the RTextTools library in `R`.  It consists of titles from NYTimes front page news and associated codes according to [Amber Boydstun's classification scheme](http://www.policyagendas.org/sites/policyagendas.org/files/Boydstun_NYT_FrontPage_Codebook_0.pdf).


```
## Loading required package: SparseM
## 
## Attaching package: 'SparseM'
## 
## The following object is masked from 'package:base':
## 
## backsolve
## 
## Loading required package: randomForest randomForest 4.6-7 Type rfNews() to
## see new features/changes/bug fixes. Loading required package: tree Loading
## required package: nnet Loading required package: tm Loading required
## package: e1071 Loading required package: class Loading required package:
## ipred KernSmooth 2.23 loaded Copyright M. P. Wand 1997-2009 Loading
## required package: caTools Loading required package: maxent Loading
## required package: Rcpp Loading required package: glmnet Loading required
## package: Matrix Loading required package: lattice
## 
## Attaching package: 'Matrix'
## 
## The following object is masked from 'package:SparseM':
## 
## det
## 
## Loaded glmnet 1.9-5
## 
## Loading required package: tau
```



```r
# Code adapted from Collingwood and Jurka see here:
# http://www.rtexttools.com/1/post/2012/02/rtexttools-short-course-materials.html
# READ THE CSV DATA from the RTextTools package Note that RTextTools has
# many dependencies, but Collingwood & Jurka [wisely] chose to keep all the
# dependencies R-friendly (read, no Java that I know of).
data(NYTimes)

# there isn't that much data in this dataset for training so we're going to
# subset down only to those data that contain a lot of observations
table(NYTimes$Topic.Code)
```

```
## 
##   1   2   3   4   5   6   7   8  10  12  13  14  15  16  17  18  19  20 
##  71  88 185  19  83  87  34  33  50 163  22  40 172 444  81  16 662 394 
##  21  24  26  27  28  29  30  31  99 
##  20  76  47  11  75 141  29  41  20
```

```r
# consider only using 3, 12, 15, 16, 19, 20, 29
valid = c(3, 12, 15, 16, 19, 20, 29)
NYTimes = NYTimes[NYTimes$Topic.Code %in% valid, ]
table(NYTimes$Topic.Code)
```

```
## 
##   3  12  15  16  19  20  29 
## 185 163 172 444 662 394 141
```

```r
num_documents = dim(NYTimes)[1]

# Examine the data
class(NYTimes)  #make sure it is a data frame object
```

```
## [1] "data.frame"
```

```r
head(NYTimes)  # Look at the first six lines or so
```

```
##   Article_ID      Date
## 1      41246  1-Jan-96
## 2      41257  2-Jan-96
## 3      41268  3-Jan-96
## 4      41279  4-Jan-96
## 6      41302  7-Jan-96
## 9      41344 11-Jan-96
##                                                                 Title
## 1       Nation's Smaller Jails Struggle To Cope With Surge in Inmates
## 2                     FEDERAL IMPASSE SADDLING STATES WITH INDECISION
## 3 Long, Costly Prelude Does Little To Alter Plot of Presidential Race
## 4        Top Leader of the Bosnian Serbs Now Under Attack From Within
## 6                     South African Democracy Stumbles on Old Rivalry
## 9                                 High Court Is Cool To Census Change
##                                      Subject Topic.Code
## 1  Jails overwhelmed with hardened criminals         12
## 2    Federal budget impasse affect on states         20
## 3 Contenders for 1996 Presedential elections         20
## 4 Bosnian Serb leader criticized from within         19
## 6         political violence in south africa         19
## 9                             census changes         20
```

```r
summary(NYTimes)  #summarize the data
```

```
##    Article_ID           Date     
##  Min.   : 5469   29-Sep-99:   2  
##  1st Qu.:19551   1-Apr-00 :   1  
##  Median :28163   1-Apr-01 :   1  
##  Mean   :27616   1-Apr-03 :   1  
##  3rd Qu.:37194   1-Apr-04 :   1  
##  Max.   :45716   1-Apr-05 :   1  
##                  (Other)  :2154  
##                                    Title                   Subject    
##  CRISIS IN THE BALKANS: THE OVERVIEW; :   6   baseball         :  10  
##  TESTING OF A PRESIDENT: THE OVERVIEW;:   5   Enron scandal    :   5  
##  INTERNATIONAL BUSINESS;              :   4   olympics         :   5  
##  STANDOFF WITH IRAQ: THE OVERVIEW;    :   3   tennis           :   4  
##  BASEBALL PLAYOFFS                    :   2   2000 campaign    :   3  
##  BASEBALL;                            :   2   baseball playoffs:   3  
##  (Other)                              :2139   (Other)          :2131  
##    Topic.Code
##  Min.   : 3  
##  1st Qu.:16  
##  Median :19  
##  Mean   :17  
##  3rd Qu.:19  
##  Max.   :29  
## 
```

```r
sapply(NYTimes, class)  #look at the class of each column
```

```
## Article_ID       Date      Title    Subject Topic.Code 
##  "integer"   "factor"   "factor"   "factor"  "integer"
```

```r
dim(NYTimes)  #Check the dimensions, rows and columns
```

```
## [1] 2161    5
```

```r

# [OPTIONAL] SUBSET YOUR DATA TO GET A RANDOM SAMPLE we don't have that much
# data, so we're going to keep it all sample_size = 500
sample_size = num_documents
NYT_sample = NYTimes[sample(1:num_documents, size = sample_size, replace = FALSE), 
    ]

out_data = data.frame(NYT_sample$Topic.Code, NYT_sample$Title)
write.csv(out_data, "../data/nyt_title_data.csv", row.names = F)

# CREATE A TERM-DOCUMENT MATRIX THAT REPRESENTS WORD FREQUENCIES IN EACH
# DOCUMENT WE WILL TRAIN ON THE Title COLUMNS NYT_dtm =
# create_matrix(data.frame(NYT_sample$Title,NYT_sample$Subject),
NYT_dtm = create_matrix(as.vector(NYT_sample$Title), language = "english", removeNumbers = TRUE, 
    stemWords = TRUE, weighting = weightTfIdf)

NYT_dtm  # Sparse Matrix object
```

```
## A document-term matrix (2161 documents, 3437 terms)
## 
## Non-/sparse entries: 12285/7415072
## Sparsity           : 100%
## Maximal term length: 16 
## Weighting          : term frequency - inverse document frequency (normalized) (tf-idf)
```

```r

# CORPUS AND CONTAINER CREATION

# choosing the right size of training/test data is a personal decision let's
# go with an 80/20 split; this is quite common
train_n = round(sample_size * 0.8)
test_n = round(sample_size * 0.2)

corpus = create_container(NYT_dtm, NYT_sample$Topic.Code, trainSize = 1:train_n, 
    testSize = (train_n + 1):sample_size, virgin = FALSE)

names(attributes(corpus))
```

```
## [1] "training_matrix"       "classification_matrix" "training_codes"       
## [4] "testing_codes"         "column_names"          "virgin"               
## [7] "class"
```

```r
paste(NYT_sample[1, ]$Title)  # original data
```

```
## [1] "Dole Courts Democrats"
```

```r
corpus@column_names[corpus@training_matrix[1]@ja]  # preprocessed data
```

```
## [1] "court"    "democrat" "dole"
```

```r

# TRAIN MODELS
models = train_models(corpus, algorithms = c("SVM", "MAXENT"))
results = classify_models(corpus, models)
analytics = create_analytics(corpus, results)

nyt_codes = read.csv("../data/nytimes_codes.csv")
test_start_index = num_documents - train_n
svm_full = data.frame(NYT_sample[1730:2161, ]$Title, results$SVM_LABEL)
maxent_full = data.frame(NYT_sample[1730:2161, ]$Title, results$MAXENTROPY_LABEL)

names(svm_full) = c("content", "code")
names(maxent_full) = c("content", "code")
svm_full = merge(svm_full, nyt_codes)
maxent_full = merge(maxent_full, nyt_codes)

# lets take a random sample of each of these and ask people to verify the
# coding

svm_mozfest = svm_full[sample(1:432, size = 100, replace = FALSE), ]
maxent_mozfest = maxent_full[sample(1:432, size = 100, replace = FALSE), ]

write.csv(svm_mozfest, "../labeling_examples/svm_mozfest.csv", row.names = FALSE)
write.csv(maxent_mozfest, "../labeling_examples/maxent_mozfest.csv", row.names = FALSE)
```




```r
# SUMMARY OF PRECISION, RECALL, F-SCORES, AND ACCURACY SORTED BY TOPIC CODE
# FOR EACH ALGORITHM
analytics@algorithm_summary
```

```
##    SVM_PRECISION SVM_RECALL SVM_FSCORE MAXENTROPY_PRECISION
## 3           0.72       0.54       0.62                 0.74
## 12          0.46       0.53       0.49                 0.53
## 15          0.52       0.39       0.45                 0.65
## 16          0.61       0.58       0.59                 0.51
## 19          0.61       0.70       0.65                 0.65
## 20          0.73       0.72       0.72                 0.71
## 29          0.80       0.80       0.80                 0.86
##    MAXENTROPY_RECALL MAXENTROPY_FSCORE
## 3               0.51              0.60
## 12              0.56              0.54
## 15              0.45              0.53
## 16              0.63              0.56
## 19              0.68              0.66
## 20              0.71              0.71
## 29              0.71              0.78
```

```r
# SUMMARY OF LABEL (e.g. TOPIC) ACCURACY analytics@label_summary RAW SUMMARY
# OF ALL DATA AND SCORING analytics@document_summary
```



```r
x = as.character(rownames(analytics@algorithm_summary))[-20]
y = analytics@algorithm_summary$SVM_RECALL[-20]
plot(x, y, type = "l", lwd = 3, main = "Support Vector Machine Topic Accuracy", 
    ylab = "Recall Accuracy", xlab = "Topic")
abline(h = 0.75, lwd = 2, col = "maroon")
text(x, y, adj = 1.2)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-41.png) 

```r

x = as.character(rownames(analytics@algorithm_summary))[-20]
y = analytics@algorithm_summary$MAXENTROPY_RECALL[-20]
plot(x, y, type = "l", lwd = 3, main = "Maximum Entropy Topic Accuracy", ylab = "Recall Accuracy", 
    xlab = "Topic")
abline(h = 0.75, lwd = 2, col = "maroon")
text(x, y, adj = 1.2)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-42.png) 


## Unsupervised learning: topic modeling


```r
library(topicmodels)

#term frequency vectors, not tf-idf vectors
n_topics = 60
NYT_dtm = create_matrix(as.vector(NYT_sample$Title), 
                         language="english", 
                         removeNumbers=FALSE, 
                         stemWords=FALSE, #only because they are short 
                         weighting=weightTf)

rowTotals = apply(NYT_dtm , 1, sum)
NYT_dtm_full  = NYT_dtm[rowTotals> 0]  

#k = length(unique(NYT_sample$Topic.Code))
lda = LDA(NYT_dtm_full, n_topics)

topic = topics(lda, 1)
topic[1]
```

```
## Dole Courts Democrats 
##                    18
```

```r

terms = terms(lda, 10)
terms
```

```
##       Topic 1    Topic 2    Topic 3     Topic 4       Topic 5    
##  [1,] "aids"     "say"      "says"      "senate"      "will"     
##  [2,] "benefit"  "military" "fbi"       "home"        "york"     
##  [3,] "clintons" "plans"    "test"      "money"       "drug"     
##  [4,] "face"     "calls"    "just"      "global"      "state"    
##  [5,] "chaos"    "changes"  "agents"    "fire"        "law"      
##  [6,] "african"  "women"    "cover"     "republicans" "air"      
##  [7,] "patients" "sell"     "charter"   "justices"    "stay"     
##  [8,] "rally"    "concern"  "terrorist" "cash"        "asserts"  
##  [9,] "donors"   "proposes" "cases"     "short"       "donations"
## [10,] "billions" "pushing"  "managed"   "1998"        "sexual"   
##       Topic 6    Topic 7     Topic 8      Topic 9   Topic 10   Topic 11 
##  [1,] "gop"      "mideast"   "clinton"    "new"     "inquiry"  "care"   
##  [2,] "attacks"  "long"      "focus"      "see"     "russian"  "health" 
##  [3,] "begins"   "medicare"  "chinas"     "seek"    "cancer"   "hussein"
##  [4,] "search"   "afghan"    "pentagon"   "help"    "iraqs"    "powell" 
##  [5,] "victims"  "milosevic" "increase"   "prison"  "files"    "arab"   
##  [6,] "cities"   "least"     "shifts"     "cells"   "sept"     "signs"  
##  [7,] "disaster" "numbers"   "forces"     "cost"    "cut"      "likely" 
##  [8,] "form"     "standoff"  "jury"       "islamic" "show"     "hopes"  
##  [9,] "halt"     "peru"      "armed"      "cell"    "faces"    "raid"   
## [10,] "gun"      "sites"     "challenges" "stem"    "officers" "behind" 
##       Topic 12     Topic 13   Topic 14    Topic 15   Topic 16    
##  [1,] "cia"        "israel"   "leader"    "battle"   "911"       
##  [2,] "sees"       "take"     "may"       "talks"    "washington"
##  [3,] "office"     "gaza"     "hospitals" "troops"   "bushs"     
##  [4,] "region"     "ties"     "russians"  "israelis" "life"      
##  [5,] "evidence"   "cabinet"  "used"      "sports"   "future"    
##  [6,] "return"     "call"     "suggests"  "doubts"   "issue"     
##  [7,] "foes"       "control"  "gulf"      "body"     "marines"   
##  [8,] "need"       "overhaul" "status"    "quit"     "killings"  
##  [9,] "technology" "pullout"  "embrace"   "alliance" "meet"      
## [10,] "rumsfeld"   "primary"  "cited"     "begin"    "governor"  
##       Topic 17        Topic 18    Topic 19    Topic 20   Topic 21  
##  [1,] "nation"        "president" "baghdad"   "seen"     "race"    
##  [2,] "challenged"    "democrats" "officials" "crime"    "time"    
##  [3,] "qaeda"         "testing"   "top"       "states"   "congress"
##  [4,] "taliban"       "rise"      "north"     "role"     "first"   
##  [5,] "playoffs"      "facing"    "korea"     "wall"     "nato"    
##  [6,] "kabul"         "reno"      "strike"    "base"     "gives"   
##  [7,] "investigation" "abuse"     "rises"     "approves" "violence"
##  [8,] "front"         "audit"     "toll"      "found"    "job"     
##  [9,] "lead"          "costs"     "shiite"    "data"     "way"     
## [10,] "giants"        "firm"      "clues"     "makes"    "calm"    
##       Topic 22   Topic 23   Topic 24  Topic 25      Topic 26    
##  [1,] "kill"     "world"    "set"     "killing"     "news"      
##  [2,] "two"      "baseball" "fight"   "afghanistan" "dole"      
##  [3,] "steps"    "series"   "death"   "tough"       "analysis"  
##  [4,] "push"     "security" "seeking" "kills"       "political" 
##  [5,] "fears"    "win"      "despite" "suicide"     "spy"       
##  [6,] "parties"  "delay"    "backs"   "bomber"      "british"   
##  [7,] "hearing"  "officer"  "gain"    "limits"      "republican"
##  [8,] "huge"     "domestic" "little"  "former"      "hearings"  
##  [9,] "career"   "joins"    "citing"  "rule"        "senator"   
## [10,] "hospital" "message"  "turmoil" "killed"      "cases"     
##       Topic 27   Topic 28      Topic 29   Topic 30  Topic 31   
##  [1,] "vote"     "leaders"     "panel"    "china"   "plan"     
##  [2,] "court"    "army"        "aide"     "goes"    "chief"    
##  [3,] "force"    "palestinian" "urges"    "allies"  "microsoft"
##  [4,] "strategy" "tied"        "rebel"    "capital" "judge"    
##  [5,] "rules"    "bin"         "pressure" "grip"    "raises"   
##  [6,] "defense"  "laden"       "sets"     "ground"  "target"   
##  [7,] "split"    "path"        "funds"    "using"   "aiding"   
##  [8,] "missile"  "put"         "agency"   "bosnia"  "attacking"
##  [9,] "counting" "bar"         "limit"    "cuts"    "turns"    
## [10,] "last"     "hamas"       "rejects"  "press"   "led"      
##       Topic 32        Topic 33   Topic 34   Topic 35      Topic 36   
##  [1,] "iraqi"         "case"     "back"     "iraq"        "europe"   
##  [2,] "billion"       "big"      "doctors"  "threats"     "ready"    
##  [3,] "now"           "deal"     "record"   "responses"   "hope"     
##  [4,] "business"      "enron"    "across"   "peace"       "terrorism"
##  [5,] "keep"          "day"      "concerns" "falls"       "blair"    
##  [6,] "international" "ethics"   "saudis"   "missiles"    "right"    
##  [7,] "post"          "near"     "moscow"   "green"       "raise"    
##  [8,] "serbs"         "gingrich" "step"     "must"        "senators" 
##  [9,] "ads"           "syria"    "shock"    "inspections" "work"     
## [10,] "announces"     "creates"  "saudi"    "critical"    "merger"   
##       Topic 37   Topic 38    Topic 39   Topic 40       Topic 41  
##  [1,] "find"     "politics"  "overview" "terror"       "campaign"
##  [2,] "final"    "finds"     "aid"      "said"         "2000"    
##  [3,] "old"      "south"     "system"   "ban"          "gore"    
##  [4,] "ends"     "questions" "public"   "team"         "finance" 
##  [5,] "official" "poll"      "starts"   "charges"      "africa"  
##  [6,] "streets"  "issues"    "abroad"   "intelligence" "hit"     
##  [7,] "races"    "choice"    "asks"     "puts"         "safe"    
##  [8,] "sniper"   "carolina"  "coach"    "star"         "memo"    
##  [9,] "best"     "buchanan"  "charged"  "penalty"      "cheneys" 
## [10,] "ballot"   "census"    "moves"    "linked"       "sub"     
##       Topic 42       Topic 43   Topic 44   Topic 45  Topic 46   Topic 47  
##  [1,] "bush"         "war"      "one"      "power"   "end"      "election"
##  [2,] "attack"       "drugs"    "nuclear"  "tobacco" "russia"   "leaves"  
##  [3,] "police"       "town"     "fighting" "market"  "chinese"  "next"    
##  [4,] "dead"         "early"    "bomb"     "trail"   "india"    "line"    
##  [5,] "kerry"        "industry" "blast"    "takes"   "pakistan" "season"  
##  [6,] "aftereffects" "avoid"    "kills"    "place"   "imf"      "giuliani"
##  [7,] "americans"    "sars"     "start"    "site"    "gold"     "millions"
##  [8,] "drive"        "school"   "olympics" "lost"    "shadow"   "risks"   
##  [9,] "shot"         "family"   "putin"    "general" "agree"    "flee"    
## [10,] "foreign"      "oil"      "stall"    "markets" "exchief"  "tribunal"
##       Topic 48  Topic 49     Topic 50    Topic 51  Topic 52  Topic 53   
##  [1,] "house"   "crisis"     "trial"     "die"     "years"   "iran"     
##  [2,] "white"   "balkans"    "kosovo"    "fall"    "gis"     "pay"      
##  [3,] "left"    "seeks"      "change"    "study"   "mexico"  "suspect"  
##  [4,] "texas"   "party"      "challenge" "disease" "wins"    "struggle" 
##  [5,] "helps"   "government" "key"       "high"    "victory" "group"    
##  [6,] "murder"  "sharon"     "fails"     "ill"     "yankees" "nations"  
##  [7,] "journal" "elections"  "accused"   "allow"   "game"    "cleric"   
##  [8,] "vast"    "costs"      "drop"      "knicks"  "fda"     "million"  
##  [9,] "asia"    "company"    "won"       "risk"    "opens"   "gains"    
## [10,] "charge"  "reviewing"  "agree"     "cause"   "hold"    "sanctions"
##       Topic 54    Topic 55  Topic 56  Topic 57   Topic 58       Topic 59  
##  [1,] "bombing"   "debate"  "bill"    "policy"   "bank"         "report"  
##  [2,] "still"     "major"   "use"     "gets"     "palestinians" "special" 
##  [3,] "aides"     "economy" "move"    "man"      "arafat"       "many"    
##  [4,] "stocks"    "warning" "bombs"   "get"      "west"         "japan"   
##  [5,] "verdict"   "open"    "chechen" "guilty"   "killed"       "children"
##  [6,] "investors" "secret"  "shows"   "station"  "israeli"      "term"    
##  [7,] "days"      "seat"    "diallo"  "federal"  "bombings"     "korean"  
##  [8,] "plot"      "problem" "food"    "backing"  "conflict"     "second"  
##  [9,] "fronts"    "share"   "plea"    "deadline" "east"         "shooting"
## [10,] "slide"     "hmo"     "stock"   "farewell" "raids"        "dont"    
##       Topic 60
##  [1,] "city"  
##  [2,] "iraqis"
##  [3,] "rebels"
##  [4,] "arms"  
##  [5,] "shift" 
##  [6,] "mets"  
##  [7,] "scene" 
##  [8,] "going" 
##  [9,] "bold"  
## [10,] "men"
```

```r

#data("AssociatedPress", package = "topicmodels")
#lda = LDA(AssociatedPress[1:1000,], control = list(alpha = 0.1), k = 20)
#lda_inf = posterior(lda, AssociatedPress[21:30,])
```

