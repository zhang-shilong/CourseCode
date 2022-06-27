<?php
    require("utils.php");
    if(!is_session_started()){
        session_start();
    }

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

    if (isset($_SESSION["username"])) {
        $username = $_SESSION["username"];

        if (!empty($_POST["test-editor-markdown-doc"])) {
            var_dump($_POST);
            $markdown_text = htmlspecialchars($_POST["test-editor-markdown-doc"]);
            $html_text = htmlspecialchars($_POST["test-editor-html-code"]);
            date_default_timezone_set("Asia/Shanghai");
            $publish_time = date("Y/m/d H:i:s", time());
            if (preg_match("/^# .+?\n/", $markdown_text, $title_match)) {
                $title = substr($title_match[0], 2, -1);
            }
            else {
                $title = "Untitled article " . $publish_time;
            }
            $read_time = (int)(strlen($markdown_text) / 500);

            $sql_insert = "
            INSERT INTO `$table_name`
            VALUES(NULL, '$username', '$title', '$markdown_text', '$html_text', '$read_time', '$publish_time');
        ";
            if ($link->query($sql_insert)) {
                echo("<script>alert('Publish successfully!');</script>");
            }
            else {
                echo("<script>alert('[Error] Fail to insert data.');</script>");
            }
        }
        else {
            echo("<script>alert('[Error] You input nothing!');</script>");
        }
    }
    else {
        echo("<script>alert('[Error] Please login before publishing new article!');</script>");
    }
    header("refresh:0; url='../articles.php'");

?>

