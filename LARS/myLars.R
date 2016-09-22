# the modified lars-lasso algorithm
 myLars <- function(X,y){ 
  n = nrow(X)
  p = ncol(X)
  # for storing beta values
  beta_mat <- matrix(0, nrow=p, ncol=12)
  # init
  b_hat = rep(0, min(n,p))
  A <- list()
  sequence <- list() # for log.
  lam = .Machine$integer.max
  user_defined_lam = 20
  count = 1
  first = TRUE
  
  while(1){
    cat(count)
    # update C_j(lam_k)
    c = rep(0, min(n,p))
    for(j in 1:p){
      c[j] = t(X[,j]) %*% (y - X %*% b_hat)
    }
    # Update Active Set
    if(first){
      first = FALSE
      max_id = which(max(abs(c)) == abs(c)) # taking the id that maximizes X_j'y
      lam_k = max(abs(c))
      sequence[length(sequence) + 1] <- max_id
      A[length(A) + 1] <- max_id # Now Active set is {1} (or {max_id}) 
      beta_mat[,count] <- b_hat
    }else{
      if(max_id > 0){
        sequence[length(sequence) + 1] <- max_id
        A[length(A) + 1] <- max_id
        beta_mat[,count] <- b_hat
      }else{
        sequence[length(sequence) + 1] <- -A[[keep_id_lam_tilda]]
        A[[keep_id_lam_tilda]] <- NULL
        beta_mat[,count] <- b_hat
      }
    }
    
    # Stopping criteria  # will implement later
    if(count == 12){break}
    
    # Construct Z matrix
    Z = matrix(0, nrow=n, ncol=length(A))
    for(i in 1 : length(A)){
      Z[,i] = sign(c[A[[i]]])*X[,A[[i]]]
    }
    v = solve(t(Z) %*% Z) %*% rep(1, length(A))
    
    
    # Update lambda
    if(length(A)!=1){ # if not first time.
      lam_k = lam
    }
    lam_hat = 0
    for(j in 1 : p){
      if( (j %in% A) == FALSE){
        alpha = lam_k*t(X[,j])%*%Z%*%v  
        gamma = t(X[,j])%*%Z%*%v
        #if(c[j] > 0){
        temp1 = (c[j] - alpha) / (1 - gamma)  
        #}else{
        temp2 = (-c[j] + alpha) / (1 + gamma) 
        if(temp1 > lam_k){ temp1 = 0}
        if(temp2 > lam_k){ temp2 = 0}
        
        if(temp1 > temp2){
          temp= temp1; flag1 = TRUE; flag2 = FALSE
        }else{
          temp = temp2; flag2 = TRUE; flag1 = TRUE
        }
        if(temp > 0 & temp > lam_hat & temp < lam_k){
            lam_hat = temp
            keep_id_lam_hat = j
        }
      }
    }
    
    
    lam_tilda = 0
    for(jj in 1 : length(A)){
      temp = lam_k + b_hat[A[[jj]]]/(v[jj]*sign(c[A[[jj]]]))
      if(temp < lam_k & temp > lam_tilda){
        lam_tilda = temp
        keep_id_lam_tilda = jj # A[[j]]
      }
    }
    
    
    # Update lamda
    if(lam_hat > lam_tilda){
      lam = lam_hat 
    }else{
      lam = lam_tilda
      print("FLAG!!!")
    }
    #lam = lam_hat
    #lam_tilda = 0
    
    
    # Update b_hat 
    for(jj in 1:length(A)){
      b_hat[A[[jj]]] = b_hat[A[[jj]]] + (lam_k - lam)*v[jj]*sign(c[A[[jj]]])
    }
    
    # Fix indexing 
    if(lam_hat > lam_tilda){ # max_id is the candidate id for a new member in Active set
      max_id = keep_id_lam_hat
    }else{
      #delete this id (keep_id_lam_tilda) from Active set
      max_id = -1
    }
    
    count = count + 1
  }

  return(list(A, as.numeric(sequence), beta_mat))
}

