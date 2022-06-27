<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <title>Resume - 张世龙的个人博客</title>
    <link href="scripts/style.css" rel="stylesheet" type="text/css">
    <script src="base/jquery-1.12.4.min.js"></script>
    <script type="text/javascript" src="scripts/func.js"></script>
</head>

<body>
    <?php
        require("scripts/utils.php");
        if(!is_session_started()){
            session_start();
        }
    ?>
    <?php include("scripts/sidebar.html"); ?>
    <?php include("scripts/header.php"); ?>
    <!-- 右侧内容区域 -->
    <div class="main-wrapper">
        <div class="pdf-wrapper">
            <object data="assets/resume.pdf" type="application/pdf" width="100%" height="600px"></object>
        </div>
    </div>

</body>
</html>
