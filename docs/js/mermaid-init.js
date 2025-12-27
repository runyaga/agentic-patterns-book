// Initialize mermaid diagrams
document.addEventListener('DOMContentLoaded', function() {
    mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        securityLevel: 'loose'
    });

    // Find pre.mermaid elements and convert them for mermaid rendering
    document.querySelectorAll('pre.mermaid').forEach(function(pre) {
        var code = pre.querySelector('code');
        if (code) {
            // Move the content from code to pre and remove code element
            pre.textContent = code.textContent;
        }
    });

    // Re-run mermaid to render the diagrams
    mermaid.run();
});
