library(here)
library(tidyverse)

# Section 1. I/O
sonar <- read.csv(here("datasets", "Sonar.csv"))
head(sonar)
head(sonar, n = 10)
colnames(sonar)
dim(sonar)
nrow(sonar)

target <- sonar$Class
class(target)
mine <- sonar %>% filter(Class == "M")

sonar_with_factors <- read.csv(here("datasets", "Sonar.csv"), stringsAsFactors = TRUE)
str(sonar_with_factors)

v1 <- sonar$V1
is.vector(v1)
length(v1)
sonar[5:10, ]
v1 <- c(v1, 1.5)
length(v1)
v1

sonar_matrix <- as.matrix(sonar)
is.matrix(sonar_matrix)
str(sonar_matrix)

# Using the subset() function and select to remove the Class column.
sonar_remove <- subset(sonar, select = -Class)
str(sonar_remove)
dim(sonar_remove)

sonar_remove1 <- subset(sonar, select = -c(V1, V2, V3))
str(sonar_remove1)

# Section 2. Numerical summary 
summary(sonar)
sonar %>% glimpse()

sd(sonar$V1)


# 按照Class 来分得到V1的中位数
aggregate(V1 ~ Class, data = sonar, FUN = median)

sonar %>% 
  ggplot(aes(x = Class, y = V1)) + 
  geom_boxplot() + 
  theme_classic() + 
  labs(x =  "Class", y = "V1 content") + 
  ggtitle("V1 Content by Class")


sonar %>% 
  ggplot(aes(x = V1, y = V2)) + 
  geom_point() + 
  geom_smooth(method = "lm", se = FALSE) + 
  labs(x =  "V1", y = "V2") +
  ggtitle("V2 versus V1 Scatter Plot")




