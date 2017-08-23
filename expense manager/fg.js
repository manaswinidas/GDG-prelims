$(document).ready(function(){
        $("#savebutton").click(function(){
            var name = $("#name").val();
            var category = $("#category").val();
            var limit = $("#limit").val();
            var amount = $("#amount").val();
            var method = $("#method").val();
            var markup = "<tr><td>" + name + "</td><td>" + category + "</td><td>" + limit + "</td><td>" + amount +"</td><td>" + method; + "</td></tr>"
            $(".table").append(markup);
        });
 });