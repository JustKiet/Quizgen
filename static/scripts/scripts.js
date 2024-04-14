const textarea = document.getElementById("textarea")

function getSelectedText() {
    var selectedText = textarea.value.substring(textarea.selectionStart, textarea.selectionEnd);
    textarea.dataset.selectedText = selectedText;
}

function f1(e){
    let value = e.value;
    var selectedText = textarea.dataset.selectedText;

}

function handleTab(event) {
    if (event.key === "Tab") {
        event.preventDefault();
        var start = textarea.selectionStart;
        var end = textarea.selectionEnd;
        textarea.value = textarea.value.substring(0, start) + "\t" + textarea.value.substring(end);
        textarea.selectionStart = textarea.selectionEnd = start + 1;
        return false;
    }
}

