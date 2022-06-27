<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <title>Articles - 张世龙的个人博客</title>
    <link href="style.css" rel="stylesheet" type="text/css">
    <script src="base/jquery-1.12.4.min.js"></script>
    <script type="text/javascript" src="index.js"></script>
    <link rel="stylesheet" href="base/editor.md-1.5.0/css/editormd.css" />
    <script src="base/editor.md-1.5.0/editormd.js"></script>
    <script type="text/javascript">
        $(function() {
            const height = String(window.innerHeight * 0.65) + "px";
            let editor = editormd("test-editor", {
                width: "100%",
                height: height,
                path: "./base/editor.md-1.5.0/lib/",
                emoji: false,
                tex: true,
                flowChart: true,
                sequenceDiagram: true,
                previewCodeHighlight: true,
                toolbarAutoFixed: true,
                toolbarIcons: "full",
                saveHTMLToTextarea: true
            });
        });
    </script>
</head>

<body>
    <?php
        require("utils.php");
        if(!is_session_started()){
            session_start();
        }
    ?>

    <!-- 左边导航栏 -->
    <div class="left-wrapper" style="background: white;">
        <ul class="article-list">
        <?php
            $db_name = "web_class";
            $table_name = "articles";
            $sql_create_table = "
                CREATE TABLE `$table_name`
                (
                    `id` INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                    `username` TEXT,
                    `title` TEXT,
                    `markdown_text` TEXT,
                    `html_text` TEXT,
                    `read_time` INT,
                    `publish_time` DATETIME
                );";
            $link = check_table($db_name, $table_name, $sql_create_table);

            $sql_query = "SELECT `id`, `title` FROM `$table_name` ORDER BY `id`;";
            $result_query = $link->query($sql_query);
            $rows = $result_query->num_rows;
            while ($row = $result_query->fetch_object()) {
                ?>
                <li class="article-line"><a class="black-ref" href="articles.php?id=<?php echo($row->id); ?>"><?php echo($row->title); ?></a></li>
                <?php
            }
        ?>
        </ul>

    </div>
    <?php include("header.php"); ?>
    <!-- 右侧内容区域 -->
    <div class="main-wrapper">
        <!-- New Article按钮 -->
        <div style="position: fixed; top: 18%; right: 10%;">
            <input type="button" value="New Article" class="simple-button" onclick="displayWriteWindow()" />
        </div>
        <!-- 博客内容部分 -->
        <div class="main-content">
            <?php
                if (!isset($_GET["id"])){
                    $id = 1;
                }
                else {
                    $id = $_GET["id"];
                }
                $sql_query = "SELECT * FROM `$table_name` WHERE `id` = '$id';";
                $result_query = $link->query($sql_query);
                while ($row = $result_query->fetch_object()) {
                    echo(htmlspecialchars_decode($row->html_text));
                    $username = $row->username;
                    $read_time = $row->read_time;
                    $publish_time = $row->publish_time;
                    echo("<script>addArticleInfo('$username', '$read_time', '$publish_time');</script>");
                }
            ?>

        </div>
        <?php include("catalog.html"); ?>
    </div>

    <!-- 展示md文本框 -->
    <div id="write-window-id" class="write-window">
        <div style="position: absolute; top: 0; right: 20px; font-size: xx-large">
            <a href="javascript:void(0)" onclick="hideWriteWindow()" class="black-ref">×</a>
        </div>
        <p style="font-size: x-large; text-align: center;">Write in markdown</p>
        <form method="post" action="submit_article.php">

            <div id="test-editor">
                <textarea id="inp-content" style="display:none"></textarea>
            </div>
            <center>
                <input type="submit" name="submit" value="提交" id="submit-button" onclick="submitButton()" style="margin-top: 10px;" />
            </center>
        </form>
    </div>
    <div id="shadow" class="shadow-window"></div>

</body>
</html>
