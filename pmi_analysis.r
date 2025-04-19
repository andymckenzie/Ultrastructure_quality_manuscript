# Read the CSV file
data <- read.csv("2025-04-18-EM Annotation Tracker.csv", header = TRUE, stringsAsFactors = FALSE)

# Clean the data - fill in missing donor IDs
for (i in 2:nrow(data)) {
  if (is.na(data$Donor.ID[i])) {
    data$Donor.ID[i] <- data$Donor.ID[i-1]
    data$PMI[i] <- data$PMI[i-1]
  }
}

# Convert to numeric values
data$AIZ.percentage..in.one.annotated.image. <- as.numeric(gsub("%", "", data$AIZ.percentage..in.one.annotated.image.))
data$PMI <- as.numeric(gsub(" hours", "", data$PMI))
data$Percentage.of.images.with.AIZ.artifacts..count.total. <- as.numeric(gsub("%.*$", "", data$Percentage.of.images.with.AIZ.artifacts..count.total.))

# Split the data by region
thalamus_data <- subset(data, Region == "Thalamus")
cortex_data <- subset(data, Region == "Cortex")

cor.test(cortex_data$AIZ.percentage..in.one.annotated.image., cortex_data$PMI, method = "spearman")
cor.test(cortex_data$Percentage.of.images.with.AIZ.artifacts..count.total., cortex_data$PMI, method = "spearman")

cor.test(thalamus_data$AIZ.percentage..in.one.annotated.image., thalamus_data$PMI, method = "spearman")
cor.test(thalamus_data$Percentage.of.images.with.AIZ.artifacts..count.total., thalamus_data$PMI, method = "spearman")

t.test(thalamus_data$Percentage.of.images.with.AIZ.artifacts..count.total., cortex_data$Percentage.of.images.with.AIZ.artifacts..count.total.)
t.test(thalamus_data$AIZ.percentage..in.one.annotated.image., cortex_data$AIZ.percentage..in.one.annotated.image.)