// Clear log window and results
print("\\Clear");
run("Clear Results");

if (roiManager("count") == 0) {
    exit("No ROIs in the ROI Manager");
}

n = roiManager("count");
row = 0;  // Keep track of row number explicitly

for (i = 0; i < n; i++) {
    roiManager("Select", i);
    // Remove the Fit Spline command and just use interpolation
    run("Interpolate", "interval=1");
    
    roiIndex = Roi.getName;
    if (roiIndex == "") roiIndex = i;
    
    getSelectionCoordinates(x, y);
    for (j = 0; j < x.length; j++) {
        setResult("ROI", row, roiIndex);
        setResult("X", row, x[j]);
        setResult("Y", row, y[j]);
        row++;
    }
}

updateResults();
print("Total points: " + row);