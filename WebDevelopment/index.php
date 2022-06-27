<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <title>张世龙的个人博客</title>
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
        <!-- 博客内容部分 -->
        <div class="main-content">
            <a name="About"></a><h2>Welcome</h2>
            <p>本网站是《基于 Linux 的 Web 开发》的课程设计。</p>
            <p>主要实现了 5 个页面，包括主页、个人简历、博客文章、出版物和关于页面。</p>
            <p>包括以下特色：</p>
            <ul>
                <li>基于 JavaScript 生成的动态目录，会根据文章的 h2 标题自动生成，并展示当前浏览的位置，且目录支持点击跳转；</li>
                <li>支持 Feedback 功能，PHP 与 MySQL 交互，将 Feedback 表单结果存入关系型数据库；</li>
                <li>支持登录和登出功能；</li>
                <li>登录后在 Articles 页面可以发表新文章，使用了开源 Markdown 编辑器 Editor.md；</li>
                <li>Articles 页面的文章被存储在 MySQL 数据库中，点击左侧的文章列表可跳转，同时支持动态目录生成；</li>
                <li>宽度自适应页面，当页面宽度不足时，自动隐藏目录。</li>
            </ul>
            <p>部分内容的实现参考了网络公开资料。</p>
            <p>本项目使用的 JavaScript 开源库：</p>
            <ul>
                <li>jQuery: <a href="https://jquery.com/" class="black-ref">https://jquery.com/</a></li>
                <li>Editor.md: <a href="https://pandao.github.io/editor.md/" class="black-ref">https://pandao.github.io/editor.md/</a></li>
            </ul>
            <a name="Special Attention"></a><h2>Special Attention</h2>
            <p>本项目的环境设置：</p>
            <ul>
                <li>MySQL 数据库默认端口：localhost:3306，用户名：root，密码：root，MySQL 服务未开启或未连接成功时会弹窗提示，该设置可通过修改 utils.php 的 check_table() 函数变量更改；</li>
                <li>MySQL 数据库的 schema 名称为 web_class，其下存在 3 张表：login、feedback 和 articles，数据表未创建时会自动创建；</li>
                <li>登录功能仅支持唯一的个人账户，用户名：zsl，密码：zslzsl。</li>
            </ul>
        </div>
        <?php include("scripts/catalog.html"); ?>
    </div>

</body>
</html>
