const fs = require('fs');
let code = fs.readFileSync('App.jsx', 'utf8');

code = code.replace(
  "  let currentTestIdx = targetTestIdxRef.current >= 0 ? targetTestIdxRef.current : -1;",
  "  let currentTestIdx = targetTestIdxRef.current >= 0 ? targetTestIdxRef.current : -1;\n  console.log(`[NAV DEBUG] Render start. targetTestIdxRef: ${targetTestIdxRef.current}, plotXRange: ${JSON.stringify(plotXRange)}`);"
);

code = code.replace(
  "      if (center < realStart || center > realEnd) {",
  "      console.log(`[NAV DEBUG] Evaluating test ${currentTestIdx}. center: ${center}, realStart: ${realStart}, realEnd: ${realEnd}`);\n      if (center < realStart || center > realEnd) {"
);

code = code.replace(
  "        currentTestIdx = -1; // will recompute below",
  "        console.log(`[NAV DEBUG] Center drifted outside! Recomputing...`);\n        currentTestIdx = -1; // will recompute below"
);

code = code.replace(
  "      currentTestIdx = foundIdx;\n      targetTestIdxRef.current = currentTestIdx;",
  "      console.log(`[NAV DEBUG] Fallback found test ${foundIdx}`);\n      currentTestIdx = foundIdx;\n      targetTestIdxRef.current = currentTestIdx;"
);

code = code.replace(
  "  const jumpToTest = (targetIdx) => {",
  "  const jumpToTest = (targetIdx) => {\n    console.log(`[NAV DEBUG] jumpToTest called with ${targetIdx}`);"
);

code = code.replace(
  "    setPlotXRange({ min: zoomStart, max: zoomEnd });",
  "    console.log(`[NAV DEBUG] jumpToTest setting plotXRange: ${zoomStart} - ${zoomEnd}`);\n    setPlotXRange({ min: zoomStart, max: zoomEnd });"
);

fs.writeFileSync('App.jsx', code);
