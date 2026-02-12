window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
  },
  options: {
    // Tell MathJax to only process elements with class "arithmatex"
    processHtmlClass: "arithmatex",
    ignoreHtmlClass: ".*"
  }
};


document$.subscribe(() => {
  MathJax.typesetPromise()
})
