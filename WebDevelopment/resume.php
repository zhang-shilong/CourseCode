<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <title>Resume - 张世龙的个人博客</title>
    <link href="style.css" rel="stylesheet" type="text/css">
    <script src="base/jquery-1.12.4.min.js"></script>
    <script type="text/javascript" src="index.js"></script>
</head>

<body>
    <?php
        require("utils.php");
        if(!is_session_started()){
            session_start();
        }
    ?>
    <?php include("sidebar.html"); ?>
    <?php include("header.php"); ?>
    <!-- 右侧内容区域 -->
    <div class="main-wrapper">
        <div class="pdf-wrapper">
            <object data="media/resume.pdf" type="application/pdf" width="100%" height="600px"></object>
        </div>
    </div>

</body>
</html>
