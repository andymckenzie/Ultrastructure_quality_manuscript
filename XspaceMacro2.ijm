// Macro to calculate the total area of ROIs from the ROI Manager
// Created on March 05, 2025

// Clear log window
if (isOpen("Log")) {
  print("\\Clear");
} else {
  run("Log");
}

// Check if ROI Manager is open
if (!isOpen("ROI Manager")) {
  showMessage("Error", "ROI Manager is not open. Please open it with your ROIs first.");
  exit();
}

// Get the number of ROIs in the ROI Manager
// Using a direct count approach instead of relying on Results table
n = roiManager("count");

// Check if there are ROIs in the manager
if (n == 0) {
  showMessage("Error", "No ROIs found in ROI Manager. Please add your selections first.");
  exit();
}

// Clear results before measuring
run("Clear Results");

// Set up measurements
run("Set Measurements...", "area mean integrated display redirect=None decimal=3");

// Get image info
title = getTitle();
width = getWidth();
height = getHeight();
getPixelSize(unit, pixelWidth, pixelHeight);
totalImageArea = width * height * pixelWidth * pixelHeight;

// Measure all ROIs in the manager
roiManager("Measure");

// Calculate total area from results
totalArea = 0;
for (i=0; i<nResults; i++) {
  area = getResult("Area", i);
  totalArea += area;
}

// Print results to log
print("===== RESULTS FOR: " + title + " =====");
print("Total number of ROIs analyzed: " + n);
print("Total area of all ROIs: " + totalArea + " square " + unit);

// Calculate percentage of total area
percentage = (totalArea / totalImageArea) * 100;
print("ROI area percentage: " + percentage + "% of total image area");

// Make sure log window is visible
selectWindow("Log");
print("Analysis complete.");