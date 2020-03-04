
df=read.table("ET0_PM_2016.csv",header=T, sep=";",as.is=T)

df$Ta_med = as.numeric(df$Ta_med)
df$Ta_max = as.numeric(df$Ta_max)
df$Ta_min = as.numeric(df$Ta_min)
df$UR_med = as.numeric(df$UR_med)
df$UR_max = as.numeric(df$UR_max)
df$UR_min = as.numeric(df$UR_min)
df$RG = as.numeric(df$RG)
df$V = as.numeric(df$V)
df$P = as.numeric(df$P)
df$ET0 = as.numeric(df$ET0)
  
sapply(df, typeof)


library(tidyr)
df = df %>% drop_na()

df <- df[, c( "Ta_med","UR_med","RG", "V" ,"P","ET0"
 )]

names(df) <- c( "T_med","RH_med","RG", "W" ,"P","ET0")

install.packages('psych')
library(psych)
pairs.panels(df[,-5], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
             )

install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")
chart.Correlation(df, histogram=TRUE, pch=19)

install.packages("brnn")

x <- data.matrix(df[,1:5])
y <- df[,6]

library(brnn)



par(mfrow=c(2,2))


brnn = brnn(x,y,neurons=2,normalize=TRUE,epochs=1000,mu=0.005,mu_dec=0.1,mu_inc=10,mu_max=1e10,min_grad=1e-10,change = 0.001,cores=1,verbose=FALSE,Monte_Carlo = FALSE,tol = 1e-06, samples = 20)

#Number of parameters (weights and biases) to estimate: 22 
#Nguyen-Widrow method
#Scaling factor= 0.7000931 
#gamma= 21.3605   alpha= 0.3439   beta= 150.9733 

brnn$Ed
# 17.19721
brnn$Ew
# 31.05256
cor(brnn$y,predict(brnn))
# 0.9785874
p1 = plot(brnn$y,predict(brnn), 
    xlab="ET0 - Penman-Monteith", ylab="ET0 - Bayesian Neural Network", main="Using 2 neurons \n and 20 Monte-Carlo samples")
text(x=2.5, y=9,labels="R²=0.9785874")

brnn = brnn(x,y,neurons=5,normalize=TRUE,epochs=1000,mu=0.005,mu_dec=0.1,mu_inc=10,mu_max=1e10,min_grad=1e-10,change = 0.001,cores=1,verbose=FALSE,Monte_Carlo = FALSE,tol = 1e-06, samples = 40)

#Number of parameters (weights and biases) to estimate: 55 
#Nguyen-Widrow method
#Scaling factor= 0.7002161 
#gamma= 51.2894   alpha= 0.564    beta= 223.0698 

brnn$Ed
# 11.57196
brnn$Ew
# 45.46829
cor(brnn$y,predict(brnn))
# 0.9856427
p2 = plot(brnn$y,predict(brnn), 
    xlab="ET0 - Penman-Monteith", ylab="ET0 - Bayesian Neural Network", main="Using 5 neurons \n and 40 Monte-Carlo samples")
text(x=2.5, y=9,labels="R²=0.9856427")

brnn = brnn(x,y,neurons=10,normalize=TRUE,epochs=2000,mu=0.005,mu_dec=0.1,mu_inc=10,mu_max=1e10,min_grad=1e-10,change = 0.001,cores=1,verbose=FALSE,Monte_Carlo = FALSE,tol = 1e-06, samples = 40)

#Number of parameters (weights and biases) to estimate: 110 
#Nguyen-Widrow method
#Scaling factor= 0.7003092 
#gamma= 104.0183          alpha= 0.1793   beta= 253.7169 



brnn$Ed
# 10.07024
brnn$Ew
# 290.0158
cor(brnn$y,predict(brnn))
# 0.9875177
p3 = plot(brnn$y,predict(brnn), 
    xlab="ET0 - Penman-Monteith", ylab="ET0 - Bayesian Neural Network", main="Using 10 neurons \n and 40 Monte-Carlo samples")
text(x=2.5, y=9,labels="R²=0.9875177")


brnn = brnn(x,y,neurons=15,normalize=TRUE,epochs=1000,mu=0.005,mu_dec=0.1,mu_inc=10,mu_max=1e10,min_grad=1e-10,change = 0.001,cores=1,verbose=FALSE,Monte_Carlo = FALSE,tol = 1e-06, samples = 40)

#Number of parameters (weights and biases) to estimate: 105 
#Nguyen-Widrow method
#Scaling factor= 0.7003637 
#gamma= 64.4924 	 alpha= 0.7338 	 beta= 122.239

brnn$Ed
# 21.06328
brnn$Ew
# 43.94514
cor(brnn$y,predict(brnn))
# 0.973709

p4 = plot(brnn$y,predict(brnn), 
    xlab="ET0 - Penman-Monteith", ylab="ET0 - Bayesian Neural Network", main="Using 15 neurons \n and 40 Monte-Carlo samples")
text(x=2.5, y=9,labels="R²=0.973709")




df_scaled <- scale(df)

X <- df_scaled[, 1:5]
train_samples <- sample(1:nrow(df_scaled), 0.8 * nrow(X))
X_train <- X[train_samples,]
X_val <- X[-train_samples,]

y <- df_scaled[, 6] %>% as.matrix()
y_train <- y[train_samples,]
y_val <- y[-train_samples,]


library(keras)

# R6 wrapper class, a subclass of KerasWrapper
ConcreteDropout <- R6::R6Class("ConcreteDropout",
  
  inherit = KerasWrapper,
  
  public = list(
    weight_regularizer = NULL,
    dropout_regularizer = NULL,
    init_min = NULL,
    init_max = NULL,
    is_mc_dropout = NULL,
    supports_masking = TRUE,
    p_logit = NULL,
    p = NULL,
    
    initialize = function(weight_regularizer,
                          dropout_regularizer,
                          init_min,
                          init_max,
                          is_mc_dropout) {
      self$weight_regularizer <- weight_regularizer
      self$dropout_regularizer <- dropout_regularizer
      self$is_mc_dropout <- is_mc_dropout
      self$init_min <- k_log(init_min) - k_log(1 - init_min)
      self$init_max <- k_log(init_max) - k_log(1 - init_max)
    },
    
    build = function(input_shape) {
      super$build(input_shape)
      
      self$p_logit <- super$add_weight(
        name = "p_logit",
        shape = shape(1),
        initializer = initializer_random_uniform(self$init_min, self$init_max),
        trainable = TRUE
      )

      self$p <- k_sigmoid(self$p_logit)

      input_dim <- input_shape[[2]]

      weight <- private$py_wrapper$layer$kernel
      
      kernel_regularizer <- self$weight_regularizer * 
                            k_sum(k_square(weight)) / 
                            (1 - self$p)
      
      dropout_regularizer <- self$p * k_log(self$p)
      dropout_regularizer <- dropout_regularizer +  
                             (1 - self$p) * k_log(1 - self$p)
      dropout_regularizer <- dropout_regularizer * 
                             self$dropout_regularizer * 
                             k_cast(input_dim, k_floatx())

      regularizer <- k_sum(kernel_regularizer + dropout_regularizer)
      super$add_loss(regularizer)
    },
    
    concrete_dropout = function(x) {
      eps <- k_cast_to_floatx(k_epsilon())
      temp <- 0.1
      
      unif_noise <- k_random_uniform(shape = k_shape(x))
      
      drop_prob <- k_log(self$p + eps) - 
                   k_log(1 - self$p + eps) + 
                   k_log(unif_noise + eps) - 
                   k_log(1 - unif_noise + eps)
      drop_prob <- k_sigmoid(drop_prob / temp)
      
      random_tensor <- 1 - drop_prob
      
      retain_prob <- 1 - self$p
      x <- x * random_tensor
      x <- x / retain_prob
      x
    },

    call = function(x, mask = NULL, training = NULL) {
      if (self$is_mc_dropout) {
        super$call(self$concrete_dropout(x))
      } else {
        k_in_train_phase(
          function()
            super$call(self$concrete_dropout(x)),
          super$call(x),
          training = training
        )
      }
    }
  )
)

# function for instantiating custom wrapper
layer_concrete_dropout <- function(object, 
                                   layer,
                                   weight_regularizer = 1e-6,
                                   dropout_regularizer = 1e-5,
                                   init_min = 0.1,
                                   init_max = 0.1,
                                   is_mc_dropout = TRUE,
                                   name = NULL,
                                   trainable = TRUE) {
  create_wrapper(ConcreteDropout, object, list(
    layer = layer,
    weight_regularizer = weight_regularizer,
    dropout_regularizer = dropout_regularizer,
    init_min = init_min,
    init_max = init_max,
    is_mc_dropout = is_mc_dropout,
    name = name,
    trainable = trainable
  ))
}


n <- nrow(X_train)
n_epochs <- 150
batch_size <- 2
output_dim <- 1
num_MC_samples <- 20
l <- 1e-4
wd <- l^2/n
dd <- 2/n

get_model <- function(input_dim, hidden_dim_1,hidden_dim_2,hidden_dim_3) {
  
  input <- layer_input(shape = input_dim)
  output <-
    input %>% layer_concrete_dropout(
      layer = layer_dense(units = hidden_dim_1, activation = "relu"),
      weight_regularizer = wd,
      dropout_regularizer = dd
    ) %>% layer_concrete_dropout(
      layer = layer_dense(units = hidden_dim_2, activation = "relu"),
      weight_regularizer = wd,
      dropout_regularizer = dd
    ) %>% layer_concrete_dropout(
      layer = layer_dense(units = hidden_dim_3, activation = "relu"),
      weight_regularizer = wd,
      dropout_regularizer = dd
    )
  
  mean <-
    output %>% layer_concrete_dropout(
      layer = layer_dense(units = output_dim),
      weight_regularizer = wd,
      dropout_regularizer = dd
    )
  
  log_var <-
    output %>% layer_concrete_dropout(
      layer_dense(units = output_dim),
      weight_regularizer = wd,
      dropout_regularizer = dd
    )
  
  output <- layer_concatenate(list(mean, log_var))
  
  model <- keras_model(input, output)
  
  heteroscedastic_loss <- function(y_true, y_pred) {
    mean <- y_pred[, 1:output_dim]
    log_var <- y_pred[, (output_dim + 1):(output_dim * 2)]
    precision <- k_exp(-log_var)
    k_sum(precision * (y_true - mean) ^ 2 + log_var, axis = 2)
  }
  
  model %>% compile(optimizer = "adam",
                    loss = heteroscedastic_loss,
                    metrics = c("mae"))
  model
}



model <- get_model(1, 100,50,25)
history <- model %>% fit(
  X_train[ ,1],
  y_train,
  validation_data = list(X_val[ , 2], y_val),
  epochs = n_epochs,
  batch_size = batch_size
)



MC_samples <- array(0, dim = c(num_MC_samples, nrow(X_val), 2 * output_dim))
for (k in 1:num_MC_samples) {
  MC_samples[k, ,] <- (model %>% predict(X_val[ ,1]))
}

means <- MC_samples[, , 1:output_dim]  
predictive_mean <- apply(means, 2, mean) 
epistemic_uncertainty <- apply(means, 2, var) 
logvar <- MC_samples[, , (output_dim + 1):(output_dim * 2)]
aleatoric_uncertainty <- exp(colMeans(logvar))


preds_1 <- data.frame(
  x1 = X_val[, 1],
  y_true = y_val,
  y_pred = predictive_mean,
  e_u_lower = predictive_mean - sqrt(epistemic_uncertainty),
  e_u_upper = predictive_mean + sqrt(epistemic_uncertainty),
  a_u_lower = predictive_mean - sqrt(aleatoric_uncertainty),
  a_u_upper = predictive_mean + sqrt(aleatoric_uncertainty),
  u_overall_lower = predictive_mean - 
                    sqrt(epistemic_uncertainty) - 
                    sqrt(aleatoric_uncertainty),
  u_overall_upper = predictive_mean + 
                    sqrt(epistemic_uncertainty) + 
                    sqrt(aleatoric_uncertainty)
)

library(ggplot2)

p1e <- ggplot(preds_1, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_1, aes(x1, y_true),color='red',shape=4)+
  geom_ribbon(aes(ymin = e_u_lower, ymax = e_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Average Temperature") + ggtitle("Epistemic Uncertainty")

p1a <- ggplot(preds_1, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_1, aes(x1, y_true),color='red',shape=4)+
  geom_ribbon(aes(ymin = a_u_lower, ymax = a_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Average Temperature") + ggtitle("Aleatoric Uncertainty")


model <- get_model(1, 100,50,25)
history <- model %>% fit(
  X_train[ ,2],
  y_train,
  validation_data = list(X_val[ , 2], y_val),
  epochs = n_epochs,
  batch_size = batch_size
)



MC_samples <- array(0, dim = c(num_MC_samples, nrow(X_val), 2 * output_dim))
for (k in 1:num_MC_samples) {
  MC_samples[k, ,] <- (model %>% predict(X_val[ ,2]))
}

means <- MC_samples[, , 1:output_dim]  
predictive_mean <- apply(means, 2, mean) 
epistemic_uncertainty <- apply(means, 2, var) 
logvar <- MC_samples[, , (output_dim + 1):(output_dim * 2)]
aleatoric_uncertainty <- exp(colMeans(logvar))




preds_2 <- data.frame(
  x1 = X_val[, 2],
  y_true = y_val,
  y_pred = predictive_mean,
  e_u_lower = predictive_mean - sqrt(epistemic_uncertainty),
  e_u_upper = predictive_mean + sqrt(epistemic_uncertainty),
  a_u_lower = predictive_mean - sqrt(aleatoric_uncertainty),
  a_u_upper = predictive_mean + sqrt(aleatoric_uncertainty),
  u_overall_lower = predictive_mean - 
                    sqrt(epistemic_uncertainty) - 
                    sqrt(aleatoric_uncertainty),
  u_overall_upper = predictive_mean + 
                    sqrt(epistemic_uncertainty) + 
                    sqrt(aleatoric_uncertainty)
)


p2e <- ggplot(preds_2, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_2, aes(x1, y_true),color='red',shape=4) +
  geom_ribbon(aes(ymin = e_u_lower, ymax = e_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Relative Humidity") 
  

p2a <- ggplot(preds_2, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_2, aes(x1, y_true),color='red',shape=4)+
  geom_ribbon(aes(ymin = a_u_lower, ymax = a_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Relative Humidity") 

model <- get_model(1, 100,50,25)
history <- model %>% fit(
  X_train[ ,3],
  y_train,
  validation_data = list(X_val[ , 3], y_val),
  epochs = n_epochs,
  batch_size = batch_size
)



MC_samples <- array(0, dim = c(num_MC_samples, nrow(X_val), 2 * output_dim))
for (k in 1:num_MC_samples) {
  MC_samples[k, ,] <- (model %>% predict(X_val[ ,3]))
}

means <- MC_samples[, , 1:output_dim]  
predictive_mean <- apply(means, 2, mean) 
epistemic_uncertainty <- apply(means, 2, var) 
logvar <- MC_samples[, , (output_dim + 1):(output_dim * 2)]
aleatoric_uncertainty <- exp(colMeans(logvar))



preds_3 <- data.frame(
  x1 = X_val[, 3],
  y_true = y_val,
  y_pred = predictive_mean,
  e_u_lower = predictive_mean - sqrt(epistemic_uncertainty),
  e_u_upper = predictive_mean + sqrt(epistemic_uncertainty),
  a_u_lower = predictive_mean - sqrt(aleatoric_uncertainty),
  a_u_upper = predictive_mean + sqrt(aleatoric_uncertainty),
  u_overall_lower = predictive_mean - 
                    sqrt(epistemic_uncertainty) - 
                    sqrt(aleatoric_uncertainty),
  u_overall_upper = predictive_mean + 
                    sqrt(epistemic_uncertainty) + 
                    sqrt(aleatoric_uncertainty)
)


p3e <- ggplot(preds_3, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_3, aes(x1, y_true),color='red',shape=4) + 
  geom_ribbon(aes(ymin = e_u_lower, ymax = e_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Wind") 
  
p3a <- ggplot(preds_3, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_3, aes(x1, y_true),color='red',shape=4)+
  geom_ribbon(aes(ymin = a_u_lower, ymax = a_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Wind")


model <- get_model(1, 100,50,25)
history <- model %>% fit(
  X_train[ ,4],
  y_train,
  validation_data = list(X_val[ , 4], y_val),
  epochs = n_epochs,
  batch_size = batch_size
)



MC_samples <- array(0, dim = c(num_MC_samples, nrow(X_val), 2 * output_dim))
for (k in 1:num_MC_samples) {
  MC_samples[k, ,] <- (model %>% predict(X_val[ ,4]))
}

means <- MC_samples[, , 1:output_dim]  
predictive_mean <- apply(means, 2, mean) 
epistemic_uncertainty <- apply(means, 2, var) 
logvar <- MC_samples[, , (output_dim + 1):(output_dim * 2)]
aleatoric_uncertainty <- exp(colMeans(logvar))


  
preds_4 <- data.frame(
  x1 = X_val[, 4],
  y_true = y_val,
  y_pred = predictive_mean,
  e_u_lower = predictive_mean - sqrt(epistemic_uncertainty),
  e_u_upper = predictive_mean + sqrt(epistemic_uncertainty),
  a_u_lower = predictive_mean - sqrt(aleatoric_uncertainty),
  a_u_upper = predictive_mean + sqrt(aleatoric_uncertainty),
  u_overall_lower = predictive_mean - 
                    sqrt(epistemic_uncertainty) - 
                    sqrt(aleatoric_uncertainty),
  u_overall_upper = predictive_mean + 
                    sqrt(epistemic_uncertainty) + 
                    sqrt(aleatoric_uncertainty)
)


p4e <- ggplot(preds_4, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_4, aes(x1, y_true),color='red',shape=4) + 
  geom_ribbon(aes(ymin = e_u_lower, ymax = e_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Solar Radiation") 
  
p4a <- ggplot(preds_4, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_4, aes(x1, y_true),color='red',shape=4)+
  geom_ribbon(aes(ymin = a_u_lower, ymax = a_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Solar Radiation") 


model <- get_model(1, 100,50,25)
history <- model %>% fit(
  X_train[ ,5],
  y_train,
  validation_data = list(X_val[ , 5], y_val),
  epochs = n_epochs,
  batch_size = batch_size
)



MC_samples <- array(0, dim = c(num_MC_samples, nrow(X_val), 2 * output_dim))
for (k in 1:num_MC_samples) {
  MC_samples[k, ,] <- (model %>% predict(X_val[ ,5]))
}

means <- MC_samples[, , 1:output_dim]  
predictive_mean <- apply(means, 2, mean) 
epistemic_uncertainty <- apply(means, 2, var) 
logvar <- MC_samples[, , (output_dim + 1):(output_dim * 2)]
aleatoric_uncertainty <- exp(colMeans(logvar))


  
preds_5 <- data.frame(
  x1 = X_val[, 5],
  y_true = y_val,
  y_pred = predictive_mean,
  e_u_lower = predictive_mean - sqrt(epistemic_uncertainty),
  e_u_upper = predictive_mean + sqrt(epistemic_uncertainty),
  a_u_lower = predictive_mean - sqrt(aleatoric_uncertainty),
  a_u_upper = predictive_mean + sqrt(aleatoric_uncertainty),
  u_overall_lower = predictive_mean - 
                    sqrt(epistemic_uncertainty) - 
                    sqrt(aleatoric_uncertainty),
  u_overall_upper = predictive_mean + 
                    sqrt(epistemic_uncertainty) + 
                    sqrt(aleatoric_uncertainty)
)




p5e <- ggplot(preds_5, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_5, aes(x1, y_true),color='red',shape=4) +
  geom_ribbon(aes(ymin = e_u_lower, ymax = e_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Precipitation") 

p5a <- ggplot(preds_5, aes(x1, y_pred), color='green') + 
  geom_point() + geom_point(data=preds_5, aes(x1, y_true),color='red',shape=4)+
  geom_ribbon(aes(ymin = a_u_lower, ymax = a_u_upper), alpha = 0.3) + ylab("Predicted Scaled ET0") + xlab("Scaled Precipitation") 


  
# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

pe <- multiplot(p1e, p2e, p3e, p4e, p5e, cols=3)

pa <- multiplot(p1a, p2a, p3a, p4a, p5a, cols=3)
