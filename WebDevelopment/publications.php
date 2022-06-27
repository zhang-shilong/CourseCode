<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <title>Publications - 张世龙的个人博客</title>
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
        <center>
            <div class="publication-wrapper">
                <p class="publication-title">A Graph-based Approach for Integrating Biological Heterogeneous Data Based on Connecting Ontology</p>
                <p class="publication-detail">IEEE BIBM 2021 | Conference article | DOI: 10.1109/BIBM52615.2021.9669700</p>
                <input type="button" value="continue" class="simple-button" onclick="openNewTab('https://ieeexplore.ieee.org/document/9669700')" />
            </div>
            <hr class="hr60">
        </center>
    </div>

</body>
</html>
