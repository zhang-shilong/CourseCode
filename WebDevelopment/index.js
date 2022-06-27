function openNewTab(page) {
    window.open(page);
}


function submitButton() {
    let $button = $("#feedback-button");
    if ($button.validate()){
        $button.addClass("disable-button");
        $button.val("提交成功");
    }
}


function generateToc() {
    let h2_list = $('h2');
    let html = "";
    for (let i=0; i < h2_list.length; i++){
        let tmp1 = h2_list[i].innerHTML.replace(/<a.+?<\/a><span.+?<\/span>/g, "");
        let tmp2 = tmp1.replace(/\s|\./g, "");
        html += "<li class='toc-line' id='" + 'toc_' + tmp2 + "'><a class='toc-ref' href='#" + tmp1 +"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + tmp1 + "</a></li>";
    }
    $('#toc ul').append(html);
}


function addArticleInfo(username, read_time, publish_time) {
    let html = "<p style='color: gray;'>作者：" + username + "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;预计阅读时间：" + read_time + "&nbsp;min&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;发表时间：" + publish_time +"</p>";
    if ($("h1").length > 0){
        $('h1').after(html);
    }
    else {
        $('h2').after(html);
    }
}


function catalogTrack() {
    let $currentHeading = null;
    for (let heading of $('h2')) {
        const $heading = $(heading);
        if ($heading.offset().top - $(document).scrollTop() > 120) {
            break;
        }
        $currentHeading = $heading;
    }

    $('.toc-line-active').removeClass('toc-line-active');
    if ($currentHeading != null) {
        const tmp = $currentHeading[0].innerHTML.replace(/\s|\.|<a.+?<\/a>|<span.+?<\/span>/g, "");
        const $current = $("#toc_" + tmp);
        $current.addClass('toc-line-active');
    }
    setTimeout(catalogTrack, 100);
}

function displayLoginWindow() {
    document.getElementById("login-window-id").style.display = "block";
    document.getElementById("shadow").style.display = "block";
}

function hideLoginWindow() {
    document.getElementById("login-window-id").style.display = "none";
    document.getElementById("shadow").style.display = "none";
}

function displayWriteWindow() {
    document.getElementById("write-window-id").style.display = "block";
    document.getElementById("shadow").style.display = "block";
}

function hideWriteWindow() {
    document.getElementById("write-window-id").style.display = "none";
    document.getElementById("shadow").style.display = "none";
}
