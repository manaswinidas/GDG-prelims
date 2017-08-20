var json = require('./dictionary.json');

$('#myTextArea').spellCheckInDialog()
$('textarea').spellAsYouType();
$(function() {$('textarea').spellAsYouType(defaultDictionary:'Espanol',checkGrammar:true);});
