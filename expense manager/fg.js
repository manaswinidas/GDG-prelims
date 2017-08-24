$(document).ready(function(){
		var myDiv =  $('.report');

        $("#savebutton").click(function(){
            var name = $("#name").val();
            var category = $("#category").val();
            var limit = $("#limit").val();
            var amount = $("#amount").val();
            var percentage = amount/limit * 100;
            console.log(percentage);
            if (percentage >= 80) {
    			error = "Warning! your expense is more than 80% of the limit";}
    		$(".report").addClass("alert alert-warning").append(error);
            var method = $("#method").val();
            var markup = "<tr><td>" + name + "</td><td>" + category + "</td><td>" + limit + "</td><td>" + amount +"</td><td>" + method; + "</td></tr>"
            $(".table").append(markup)
            if (amount >= limit) {
             $(".new").addClass("alert alert-warning")	
            }
        });
 });