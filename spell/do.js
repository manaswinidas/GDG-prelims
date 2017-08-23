$(() => {
    let dict;
    $.getJSON('dictionary.json', function(data){
      dict = data.commonWords;
    });
    $("#check").click(() => {
      var text = $('#myText').val().split(' ');
      const result = [];
      for (let x of text){
        if(dict.indexOf(x.toLowerCase()) == -1){
          result.push(`<span class="err">${x} </span>`);
        }
        else{
          result.push(x+' ');
        }
      }
      $('#paragraph').html(result.join(' '));
    });

    $('#addWord').click(() => {
      dict.push($('#word').val().trim().toLowerCase());
    });
});
