require(tm)
require(tm.corpus.Reuters21578)
library(e1071)
library(slam)
library(topicmodels)
library(randomForest)
library(SnowballC)
library(doParallel)
library(mclust)
library(cluster)

registerDoParallel(cores=4)

# Load the data
data(Reuters21578)
rt <- Reuters21578

# Split data into train/test sets
traindata <- tm_filter(rt, FUN=sFilter, "LEWISSPLIT == 'TRAIN'")
testdata <- tm_filter(rt, FUN=sFilter, "LEWISSPLIT == 'TEST'")

# Preprocess and form document-term matrix
control <- list(stemming = TRUE, stopwords = TRUE, minWordLength = 3, removeNumbers = TRUE, removePunctuation = TRUE)
trainmatrix_tf <- DocumentTermMatrix(traindata, control)

# Select the vocabulary according to TF-IDF
dim(trainmatrix_tf)
# [1] 14668 32666
summary(col_sums(trainmatrix_tf))
# Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# 1.00     1.00     2.00    32.02     7.00 35750.00
term_tfidf <- tapply(trainmatrix_tf$v/row_sums(trainmatrix_tf)[trainmatrix_tf$i], trainmatrix_tf$j, mean) * log2(nDocs(trainmatrix_tf)/col_sums(trainmatrix_tf > 0))
summary(term_tfidf)
#     Min.  1st Qu.   Median     Mean  3rd Qu.     Max.
# 0.005086 0.076140 0.131900 0.194300 0.247100 4.325000
trainmatrix_tf <- trainmatrix_tf[,term_tfidf >= 0.1]
trainmatrix_tf_rows <- row_sums(trainmatrix_tf) > 0

# Get unigram and bigram matrices
BigramTokenizer <- function(x) RWeka::NGramTokenizer(x, RWeka::Weka_control(min = 1, max = 2))
trainmatrix_bigrams <- DocumentTermMatrix(traindata, c(control, weighting = weightBin, tokenize = BigramTokenizer))
trainmatrix_bigrams <- removeSparseTerms(trainmatrix_bigrams, 0.9)
trainmatrix_bigrams_rows <- row_sums(trainmatrix_bigrams) > 0

# Use intersection of row selections
trainmatrix_rows <- trainmatrix_tf_rows & trainmatrix_bigrams_rows
trainmatrix_tf <- trainmatrix_tf[trainmatrix_rows,]
summary(col_sums(trainmatrix_tf))
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#    0.00    1.00    2.00   13.87    6.00 5783.00
dim(trainmatrix_tf)
# [1] 12849 20606
trainmatrix_bigrams <- trainmatrix_bigrams[trainmatrix_rows,]

# Fit a topic model using VEM
TM <- LDA(trainmatrix_tf, k = 30, control = list(seed = 2010))
slot(TM, "alpha")
# [1] 0.02893121
trainTopics <- posterior(TM)$topics

# Use posterior topic distribution for classification
trainAllFeatures <- cbind(as.matrix(trainmatrix_bigrams), trainTopics)

# Extracts a column of a corpus as a vector
corpusToVector <- function(corpus) {
    v <- vector()
    for (i in 1:length(corpus)) {
        v[i] <- corpus[[i]]
    }
    return(v)
}

# Using the list of top 10 topics, build vectors of whether the stories contain the topic or not
topics <- c('earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn')
topicfuncs <- lapply(topics, function(topic) { topic <- topic; function(x) if (topic %in% meta(x, "Topics")) 1 else 0 })
trainTopicClass <- lapply(topicfuncs, function(f) factor(corpusToVector(tm_map(traindata, f))[trainmatrix_rows]))

# Form test set with only terms kept in the training set
testmatrix_tf <- DocumentTermMatrix(testdata, c(control, dictionary = list(Terms(trainmatrix_tf))))
testmatrix_tf_rows <- row_sums(testmatrix_tf) > 0
testmatrix_bigrams <- DocumentTermMatrix(testdata, c(control, dictionary = list(Terms(trainmatrix_bigrams)), weighting = weightBin, tokenize = BigramTokenizer))
testmatrix_bigrams_rows <- row_sums(testmatrix_bigrams) > 0

# Use intersection of row selections
testmatrix_rows <- testmatrix_tf_rows & testmatrix_bigrams_rows
testmatrix_tf <- testmatrix_tf[testmatrix_rows,]
testmatrix_bigrams <- testmatrix_bigrams[testmatrix_rows,]

# Fit existing topic model to test data
testTM <- LDA(testmatrix_tf, model = TM)
testTopics <- posterior(testTM)$topics

# Test features and classes
testAllFeatures <- cbind(as.matrix(testmatrix_bigrams), testTopics)
testTopicClass <- lapply(topicfuncs, function(f) factor(corpusToVector(tm_map(testdata, f))[testmatrix_rows]))

# Train classifiers using training set
train <- function(features) {
    nb <- list()
    svm <- list()
    rf <- list()
    for (i in 1:length(trainTopicClass)) {
        cat('Topic class',i,'\n')
        cat('Training N-B...\n')
        nb[[i]] <- naiveBayes(features, trainTopicClass[[i]])
        cat('Training SVM...\n')
        svm[[i]] <- svm(features, trainTopicClass[[i]])
        cat('Training RF...\n')
        rf[[i]] <- foreach(ntree=rep(125, 4), .combine=combine, .packages='randomForest') %dopar%
            randomForest(features, trainTopicClass[[i]], ntree=ntree)
    }
    return(list(nb,svm,rf))
}

# Evaluate classifiers on a test set
evaluate <- function(model, features, topicClass) {
    tab <- list()
    for (i in 1:length(topicClass)) {
        results <- predict(model[[i]], features)
        tab[[i]] <- table(topicClass[[i]], results)
    }
    return(tab)
}

evaluateAll <- function(models, features, topicClass) {
    nb <- models[[1]]
    svm <- models[[2]]
    rf <- models[[3]]
    nbtab <- evaluate(nb, features, topicClass)
    svmtab <- evaluate(svm, features, topicClass)
    rftab <- evaluate(rf, features, topicClass)
    return(list(nbtab,svmtab,rftab))
}

model_bigrams <- train(as.matrix(trainmatrix_bigrams))
tab_bigrams <- evaluateAll(model_bigrams, as.matrix(trainmatrix_bigrams), trainTopicClass)
model_topics <- train(trainTopics)
tab_topics <- evaluateAll(model_topics, trainTopics, trainTopicClass)
model_all <- train(trainAllFeatures)
tab_all <- evaluateAll(model_all, trainAllFeatures, trainTopicClass)

# Evaluate classifiers on test set
test_tab_all <- evaluateAll(model_all, testAllFeatures, testTopicClass)

getAccuracy <- function(tab) {
    return((tab[1,1]+tab[2,2]) / sum(tab))
}

getRecall <- function(tab) {
    return(tab[2,2] / (tab[2,2]+tab[2,1]))
}

getPrecision <- function(tab) {
    return(tab[2,2] / (tab[2,2]+tab[1,2]))
}

getIndividualMeasures <- function(tab) {
    f <- function(tab) {
        df <- data.frame(matrix(ncol = 3, nrow = 10))
        rownames(df) <- topics
        colnames(df) <- c("Acc", "Rc", "Pc")
        # Iterate over topic classes e.g. earn, acq
        for (i in 1:length(tab)) {
            df[i,1] <- getAccuracy(tab[[i]])
            df[i,2] <- getRecall(tab[[i]])
            df[i,3] <- getPrecision(tab[[i]])
        }
        return(df)
    }
    return(list(f(tab[[1]]),f(tab[[2]]),f(tab[[3]])))
}

getIndividualMeasures(tab_bigrams)
# [[1]]
#                Acc        Rc         Pc
# earn     0.8976574 0.8272425 0.72571244
# acq      0.5412094 0.9435484 0.19459459
# money-fx 0.6121099 0.9783080 0.08313364
# grain    0.7266713 0.9012658 0.09297467
# crude    0.4450152 0.9031339 0.04275695
# trade    0.5787999 0.9881306 0.05800383
# interest 0.6050276 0.9826990 0.05304445
# ship     0.5949101 0.9218750 0.03297932
# wheat    0.6811425 0.9595960 0.04440290
# corn     0.7114951 0.9250000 0.03851158
# 
# [[2]]
#                Acc         Rc        Pc
# earn     0.9817106 0.93835364 0.9739464
# acq      0.9533038 0.62634409 0.9549180
# money-fx 0.9901938 0.75271150 0.9665738
# grain    0.9741614 0.15949367 1.0000000
# crude    0.9803098 0.27920228 1.0000000
# trade    0.9910499 0.66468843 0.9911504
# interest 0.9924508 0.73702422 0.9102564
# ship     0.9859911 0.06250000 1.0000000
# wheat    0.9850572 0.03030303 1.0000000
# corn     0.9885594 0.08125000 1.0000000
# 
# [[3]]
#                Acc        Rc        Pc
# earn     0.9965756 0.9970469 0.9868469
# acq      0.9973539 0.9778226 0.9993132
# money-fx 0.9989104 0.9783080 0.9912088
# grain    0.9987548 0.9594937 1.0000000
# crude    0.9992996 0.9743590 1.0000000
# trade    0.9991439 0.9673591 1.0000000
# interest 0.9982100 0.9411765 0.9784173
# ship     0.9987548 0.9166667 1.0000000
# wheat    0.9992217 0.9494949 1.0000000
# corn     0.9993774 0.9500000 1.0000000
# 
getIndividualMeasures(tab_topics)
# [[1]]
#                Acc        Rc         Pc
# earn     0.8904973 0.8752307 0.68924419
# acq      0.6135108 0.8413978 0.20929455
# money-fx 0.6879135 0.8568330 0.09103480
# grain    0.6573274 0.8936709 0.07488333
# crude    0.7649623 0.8974359 0.09548348
# trade    0.6068955 0.8219585 0.05258162
# interest 0.7105611 0.7993080 0.05935252
# ship     0.5844813 0.9010417 0.03149463
# wheat    0.6561600 0.9040404 0.03910004
# corn     0.5637793 0.9312500 0.02594463
# 
# [[2]]
#                Acc         Rc        Pc
# earn     0.9563390 0.83684016 0.9501257
# acq      0.8895634 0.06854839 0.7555556
# money-fx 0.9694918 0.20824295 0.7804878
# grain    0.9720601 0.20506329 0.6428571
# crude    0.9810102 0.48148148 0.7316017
# trade    0.9737723 0.00000000       NaN
# interest 0.9775080 0.00000000       NaN
# ship     0.9850572 0.00000000       NaN
# wheat    0.9845902 0.00000000       NaN
# corn     0.9875477 0.00000000       NaN
# 
# [[3]]
#                Acc        Rc        Pc
# earn     0.9986769 0.9992617 0.9944893
# acq      0.9997665 0.9979839 1.0000000
# money-fx 0.9992996 0.9934924 0.9870690
# grain    0.9998443 0.9949367 1.0000000
# crude    0.9996887 0.9914530 0.9971347
# trade    0.9991439 0.9673591 1.0000000
# interest 0.9989104 0.9757785 0.9757785
# ship     0.9997665 0.9843750 1.0000000
# wheat    0.9999222 0.9949495 1.0000000
# corn     0.9999222 0.9937500 1.0000000
# 
getIndividualMeasures(tab_all)
# [[1]]
#                Acc        Rc         Pc
# earn     0.9375049 0.8486526 0.85401189
# acq      0.6310997 0.9233871 0.22900000
# money-fx 0.7606039 0.9305857 0.12352433
# grain    0.8370301 0.9265823 0.15055533
# crude    0.7182660 0.9430199 0.08420249
# trade    0.6900148 0.9762611 0.07644052
# interest 0.7588917 0.9342561 0.08062108
# ship     0.6981866 0.9270833 0.04403761
# wheat    0.8065997 0.9444444 0.07027433
# corn     0.7692427 0.9437500 0.04859994
#   
# [[2]]
#                Acc        Rc        Pc
# earn     0.9846681 0.9509044 0.9757576
# acq      0.9630321 0.7237903 0.9439089
# money-fx 0.9916725 0.7982646 0.9633508
# grain    0.9879368 0.6278481 0.9687500
# crude    0.9919838 0.7207977 0.9806202
# trade    0.9917503 0.6913947 0.9914894
# interest 0.9920616 0.7301038 0.8978723
# ship     0.9904273 0.3593750 1.0000000
# wheat    0.9893377 0.3131313 0.9841270
# corn     0.9905051 0.2375000 1.0000000
#   
# [[3]]
#                Acc        Rc        Pc
# earn     0.9988326 1.0000000 0.9944934
# acq      1.0000000 1.0000000 1.0000000
# money-fx 0.9992996 0.9869848 0.9934498
# grain    0.9998443 0.9949367 1.0000000
# crude    0.9997665 0.9943020 0.9971429
# trade    0.9992996 0.9732938 1.0000000
# interest 0.9990661 0.9757785 0.9825784
# ship     0.9998443 0.9895833 1.0000000
# wheat    0.9999222 0.9949495 1.0000000
# corn     1.0000000 1.0000000 1.0000000
#   
getIndividualMeasures(test_tab_all)
# [[1]]
#                Acc        Rc         Pc
# earn     0.9006234 0.9253589 0.67575122
# acq      0.6371470 0.9143302 0.23377141
# money-fx 0.7975798 0.9366197 0.10830619
# grain    0.8676201 0.9477612 0.15083135
# crude    0.7565090 0.9565217 0.10440678
# trade    0.7337734 0.9464286 0.06829897
# interest 0.7873121 0.9215686 0.07544141
# ship     0.7524752 0.8823529 0.05300353
# wheat    0.8436010 0.9393939 0.06805708
# corn     0.8236157 0.8541667 0.04116466
# 
# [[2]]
#                Acc         Rc        Pc
# earn     0.9270260 0.94449761 0.7437830
# acq      0.9132747 0.43613707 0.7161125
# money-fx 0.9794646 0.35211268 0.7142857
# grain    0.9812981 0.39552239 0.7162162
# crude    0.9781812 0.37267081 0.7692308
# trade    0.9811148 0.18750000 0.6363636
# interest 0.9856986 0.32352941 0.7857143
# ship     0.9845985 0.01176471 1.0000000
# wheat    0.9891823 0.12121212 0.8888889
# corn     0.9911991 0.02083333 0.5000000
# 
# [[3]]
#                Acc         Rc        Pc
# earn     0.9303264 0.93588517 0.7575523
# acq      0.9026403 0.22741433 0.8066298
# money-fx 0.9778144 0.22535211 0.7441860
# grain    0.9778144 0.11194030 0.8823529
# crude    0.9728640 0.09316770 0.8823529
# trade    0.9807481 0.08928571 0.7692308
# interest 0.9845985 0.22549020 0.8214286
# ship     0.9844151 0.00000000       NaN
# wheat    0.9878988 0.01515152 0.5000000
# corn     0.9911991 0.00000000       NaN
# 

getAveragedMeasures <- function(tab) {
    f <- function(tab) {
        df <- data.frame(matrix(ncol = 3, nrow = 2))
        rownames(df) <- c("Macro", "Micro")
        colnames(df) <- c("Acc", "Rc", "Pc")
        sumAcc <- 0
        sumRc <- 0
        sumPc <- 0
        sumAccN <- 0
        sumAccD <- 0
        sumRcN <- 0
        sumRcD <- 0
        sumPcN <- 0
        sumPcD <- 0
        N <- 0
        # Iterate over topic classes e.g. earn, acq
        for (i in 1:length(tab)) {
            t <- tab[[i]]
            sumAcc <- sumAcc + getAccuracy(t)
            sumRc <- sumRc + getRecall(t)
            sumPc <- sumPc + getPrecision(t)
            sumAccN <- sumAccN + (t[1,1]+t[2,2])
            sumAccD <- sumAccD + (sum(t))
            sumRcN <- sumRcN + (t[2,2])
            sumRcD <- sumRcD + (t[2,2]+t[2,1])
            sumPcN <- sumPcN + (t[2,2])
            sumPcD <- sumPcD + (t[2,2]+t[1,2])
            N <- N + 1
        }
        df[1,1] <- sumAcc / N
        df[1,2] <- sumRc / N
        df[1,3] <- sumPc / N
        df[2,1] <- sumAccN / sumAccD
        df[2,2] <- sumRcN / sumRcD
        df[2,3] <- sumPcN / sumPcD
        return(df)
    }
    return(list(f(tab[[1]]),f(tab[[2]]),f(tab[[3]])))
}

getAveragedMeasures(tab_bigrams)
# [[1]]
#             Acc        Rc        Pc
# Macro 0.6394038 0.9330799 0.1366114
# Micro 0.6394038 0.8968085 0.1144603
# 
# [[2]]
#             Acc        Rc        Pc
# Macro 0.9822788 0.4331871 0.9796845
# Micro 0.9822788 0.6762918 0.9680226
# 
# [[3]]
#             Acc        Rc        Pc
# Macro 0.9985602 0.9611727 0.9955786
# Micro 0.9985602 0.9790274 0.9927570
# 
getAveragedMeasures(tab_topics)
# [[1]]
#             Acc        Rc        Pc
# Macro 0.6736088 0.8722167 0.1368414
# Micro 0.6736088 0.8655015 0.1218234
# 
# [[2]]
#            Acc        Rc        Pc
# Macro 0.967694 0.1800176       NaN
# Micro 0.967694 0.4126140 0.9046984
# 
# [[3]]
#             Acc        Rc        Pc
# Macro 0.9994941 0.9893340 0.9954472
# Micro 0.9994941 0.9945289 0.9955880
# 
getAveragedMeasures(tab_all)
# [[1]]
#            Acc        Rc        Pc
# Macro 0.760744 0.9298023 0.1761268
# Micro 0.760744 0.8987842 0.1643234
# 
# [[2]]
#             Acc        Rc        Pc
# Macro 0.9873375 0.6153110 0.9705876
# Micro 0.9873375 0.7803951 0.9657702
# 
# [[3]]
#             Acc        Rc        Pc
# Macro 0.9995875 0.9909829 0.9967664
# Micro 0.9995875 0.9955927 0.9963498
# 
getAveragedMeasures(test_tab_all)
# [[1]]
#             Acc        Rc        Pc
# Macro 0.7900257 0.9224502 0.1579033
# Micro 0.7900257 0.9247142 0.1724113
# 
# [[2]]
#             Acc        Rc        Pc
# Macro 0.9711038 0.3165780 0.7470595
# Micro 0.9711038 0.5888845 0.7370498
# 
# [[3]]
#             Acc        Rc        Pc
# Macro 0.9690319 0.1923687       NaN
# Micro 0.9690319 0.4808829 0.7663317
# 

# Combine features and classes from train and test
allFeatures <- rbind(trainAllFeatures, testAllFeatures)
topicClass <- list()
for (i in 1:length(trainTopicClass))
    topicClass[[i]] <- unlist(list(trainTopicClass[[i]], testTopicClass[[i]]))

# Clustering
diss <- dist(allFeatures)
cluster1 <- kmeans(allFeatures, 10)
cluster2 <- hclust(diss)
cluster2b <- cutree(cluster2, k = 10)
cluster3 <- Mclust(allFeatures)
sil1 <- silhouette(cluster1$cluster, diss)
sil2 <- silhouette(cluster2b, diss)
sil3 <- silhouette(cluster3$cluster, diss)
