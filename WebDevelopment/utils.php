<?php
    function is_session_started(): bool {
        if ( php_sapi_name() !== 'cli' ) {
            if ( version_compare(phpversion(), '5.4.0', '>=') ) {
                return session_status() === PHP_SESSION_ACTIVE;
            } else {
                return !(session_id() === '');
            }
        }
        return FALSE;
    }

    function check_database($db_name, $hostname, $username, $password) {
        try {
            $link = new mysqli($hostname, $username, $password);
        } catch(Exception) {
            echo("<script>alert('[Error] MySQL sever is down or wrong password!')</script>");
            die();
        }
        if ($link->connect_error) {
            echo("<script>alert('[Error] Fail to connect database!')</script>");
            die();
        }
        mysqli_query($link, "SET names UTF8");
        $link->select_db($db_name) or die("[Error] Database doesn't exist!");
        return $link;
    }

    function check_table($db_name, $table_name, $sql_create_table, $hostname="localhost:3306", $username="root", $password="root") {
        $link = check_database($db_name, $hostname, $username, $password);

        $sql_check_table = "SHOW TABLES LIKE '$table_name'";
        $res_check_table = $link->query($sql_check_table);
        if ($res_check_table->num_rows == 0) {
            $link->query($sql_create_table);
        }
        return $link;
    }

?>
