<?php
    require("utils.php");
    if(!is_session_started()){
        session_start();
    }

    if (isset($_SESSION["username"])) {
        session_unset();
        session_destroy();

        echo("<script>history.go(-1);</script>");
    }
?>
