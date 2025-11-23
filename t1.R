library(readr)
library(dplyr)
library(ggplot2)
library(here)

# 1. Read
here()
cereal <- read.csv(here('datasets', 'Cereal.csv'))
cereal %>% glimpse()
cereal %>% summary()

cereal %>% head(7)
cereal %>% class()
cereal %>% colnames()
cereal %>% nrow()
cereal %>% dim()

# 2. Data frames
# Extract the calories column
# vector
cal <- cereal$calories
# dataframe
cal <- cereal %>% select(calories)
# vector
cal <- cereal %>% pull(calories)
# Extract rows 1 to 10 from the cereal data frame.
cereal[1:10, ]
cereal %>% slice(1:10)

# Filter the cereal data frame to include only rows where mfr is "K" and assign it to Kelloggs.
kelloggs <- cereal %>% filter(mfr == "K")

# 3. factors
cereal.with.factors <- read.csv(here('datasets', 'Cereal.csv'), stringsAsFactors = TRUE)
str(cereal.with.factors) # only characters become factors
str(cereal)
levels(cereal.with.factors$mfr)
nlevels(cereal.with.factors$mfr)
str(cereal.with.factors$mfr)

# 4. vectors
cereal.cal <- cereal %>% select(calories) %>% pull()
cereal.cal %>% length()
cereal.cal[5:10]
cereal.cal <- c(cereal.cal, 1.0)
length(cereal.cal)

# 5. matrix
cereal.matrix <- as.matrix(cereal)
is.matrix(cereal.matrix)
cereal.removed <- cereal %>% select(-c(mfr, name, type))
cereal.removed |> colnames()
# Convert the 'cereal.removed' data frame to a numeric matrix.
cereal.numeric.matrix <- as.matrix(cereal.removed)
str(cereal.numeric.matrix)

summary(cereal$sodium)
mean.sodiums <- aggregate(sodium ~ mfr, data = cereal, FUN = mean)
mean.sodiums


# graphs
boxplot(sodium ~ mfr, data = cereal,
        xlab = "Manufacturer", ylab = "Sodium content", main = "Sodium Content by Manufacturer")

cereal %>% 
  ggplot(aes(x= mfr, y = sodium))+
  geom_boxplot() + 
  theme_classic() + 
  labs(x =  "Manufacturer", y = "Sodium content") + 
  ggtitle("Sodium Content by Manufacturer")


plot(calories ~ sodium, data = cereal, main = "Calories versus Sodium Scatter Plot")
cereal %>% 
  ggplot(aes(x = sodium, y = calories)) + 
  geom_point() + 
  geom_smooth(method = "lm", se = FALSE) + 
  labs(x =  "Sodium", y = "Carlories") + 
  ggtitle("Calories versus Sodium Scatter Plot")

# write to file
write.csv(kelloggs, here("datasets", "kelloggs.csv"))
