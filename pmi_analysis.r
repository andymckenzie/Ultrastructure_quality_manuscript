# Read the CSV file
data <- read.csv("2025-04-09-EM Annotation Tracker.csv", header = TRUE, stringsAsFactors = FALSE)

# Clean the data - fill in missing donor IDs
for (i in 2:nrow(data)) {
  if (is.na(data$Donor.ID[i])) {
    data$Donor.ID[i] <- data$Donor.ID[i-1]
    data$PMI[i] <- data$PMI[i-1]
  }
}

# Convert to numeric values
data$AIZ...in.one.image <- as.numeric(gsub("%", "", data$AIZ...in.one.image))
data$PMI <- as.numeric(gsub(" hours", "", data$PMI))

# Split the data by region
thalamus_data <- subset(data, Region == "Thalamus")
cortex_data <- subset(data, Region == "Cortex")

cor.test(cortex_data$AIZ...in.one.image, cortex_data$PMI, method = "spearman")
cor.test(cortex_data$Average.quality.score.grade, cortex_data$PMI, method = "spearman")

cor.test(thalamus_data$AIZ...in.one.image, thalamus_data$PMI, method = "spearman")
cor.test(thalamus_data$Average.quality.score.grade, thalamus_data$PMI, method = "spearman")

t.test(thalamus_data$Average.quality.score.grade, cortex_data$Average.quality.score.grade)