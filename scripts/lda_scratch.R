### Basic LDA Topic Model Simulation ###
### Generate Simulated Corpus ###
library(ggplot2)
library(tm)
library(MCMCpack)

simulateCorpus <- function(
  M, # number of documents
  nTerms, 
  docLengths, 
  K,  	# Number of Topics
  alphA, 	# parameter for symmetric 
  # Document/Topic dirichlet distribution
  betA, 	# parameter for Topic/Term dirichlet distribution
  Alpha=rep(alphA,K), # number-of-topics length vector 
  # set to symmetric alpha parameter 
  # across all topics
  Beta=rep(betA,nTerms))  # number-of-terms length vector 
# set to symmetric beta parameter 
# across all terms
{
  # Labels
  Terms <- paste("Term",seq(nTerms))
  Topics <- paste("Topic", seq(K))
  Documents <- paste("Document", seq(M))
  
  ## Generate latent topic and term distributions
  # "True" Document/Topic distribution matrix
  Theta <- rdirichlet(M, Alpha) 
  colnames(Theta) <- Topics
  rownames(Theta) <- Documents
  
  # "True" Topic/Term Distribution Matrix
  Phi <- rdirichlet(K, Beta) 
  colnames(Phi) <- Terms
  rownames(Phi) <- Topics
  
  ## Function to generate individual document
  generateDoc <- function(docLength, topic_dist, terms_topics_dist){
    # docLength is specific document length
    # topic_dist is specific topic distribution for this document
    # terms_topics_dist is terms distribution matrix over all topics
    document <- c()
    for (i in seq(docLength)){
      # For each word in a document, 
      # choose a topic from that 
      # document's topic distribution
      topic <- rmultinom(1, 1, topic_dist) 
      
      # Then choose a term from that topic's term distribution
      term <- rmultinom(1, 1, terms_topics_dist[topic,]) 
      
      # and append term to document vector
      document <- c(document, 
                    colnames(terms_topics_dist)[which.max(term)]) 
    }
    return(document)
  }
  
  ## generate "observed" corpus as list of terms
  corpus <- list()
  for (i in seq(M)){
    corpus[[i]] <- generateDoc(docLengths[i], Theta[i,], Phi)
  }
  
  ## convert document term vectors to frequency vectors
  freqsLists <- llply(corpus, table)
  
  ## write values to termFreqMatrix
  termFreqMatrix <- matrix(nrow=M, ncol=nTerms, 0)
  colnames(termFreqMatrix) <- Terms
  rownames(termFreqMatrix) <- Documents
  for (i in seq(M)){
    termFreqMatrix[i,names(freqsLists[[i]])] <- freqsLists[[i]]
  }
  
  stopifnot(rowSums(termFreqMatrix) == docLengths)
  
  return(list("docs"=corpus, 
              'termFreqMatrix'=termFreqMatrix, 
              "Theta"=Theta, 
              "Phi"=Phi))
  
}