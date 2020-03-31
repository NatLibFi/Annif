var set_locale_to = function(locale) {

  if (locale) {

    $.i18n().locale = locale;

  }

  $('body').i18n();
      
	$('#splash').text($.i18n('splash'));
    $('#nav-about').text($.i18n('nav-about'));
	$('#nav-feedback').text($.i18n('nav-feedback'));
	$('#api-title').text($.i18n('api-title'));
	$('#text-box-label-text').text($.i18n('text-box-label-text'));
	$('#nav-link').text($.i18n('nav-link'));
    $('#limit').text($.i18n('limit'));	
	$('#label-for-project').text($.i18n('label-for-project'));
	$('#footer').text($.i18n('footer'));
	$('#api').text($.i18n('api'));
	$('#api-h2').text($.i18n('api-h2'));
	$('#get-suggestions').text($.i18n('get-suggestions'));
	$('#suggestions').text($.i18n('suggestions'));
	$('#annif').text($.i18n('annif'));

};

jQuery(function() {

  $.i18n().load( {

   'en': {	"api-title": "API service",
	"nav-about": "About",
	"nav-feedback": "Feedback",
	"annif": "Read more.",
	"splash": "Finto AI will suggest subjects for a given text. It's based on Annif, a tool for automated subject indexing.",
    "api-h2":"API Service",
	"api": "Finto AI is also an API service that can be integrated to other systems.",
	"text-box-label-text":"Submit text to be indexed below",
	"nav-link":"Subject indexing",	
    "limit":"Number of suggestions",		
    "label-for-project": "Vocabulary and language",
	"get-suggestions": "Get suggestions",
	"suggestions": "Suggestions",
	"footer": "The data submitted via the above form or the API will not be saved anywhere. The use of the service is being monitored for development purposes."

},
	'sv': {	"api-title": "API service",
	"nav-about": "Information",
	"nav-feedback": "Respons",
	"annif": "Läs mer",
	"splash": "Finto AI föreslår deskriptorer / indexeringstermer / ämnesord för text. Det är baserat på Annif, som är ett verktyg för automatiskt indexering.",
    "api-h2":"API Service",
	"api": "Finto AI är också ett API service som man kan integrera med sina egna system.",
	"text-box-label-text":"Text",
	"nav-link":"Innehållsbeskrivning / ämnesordsindexering",	
    "limit":"Antal föreslag",		
    "label-for-project": "Volabulär och språk",
	"get-suggestions": "Analysera",
	"suggestions": "Föreslag",
	"footer": "Uppgifterna som skickas via ovanstående formuläret eller API kommer inte att sparas eller lagras. Användningen av tjänsten övervakas för utvecklingsändamål."
},

	'fi': {	"api-title": "API service",
	"nav-about": "Lisätietoa",
	"nav-feedback": "Anna palautetta",
	"annif": "Lisätietoa",
	"splash": "Finto AI ehdottaa tekstille sopivia aiheita. Palvelu perustuu Annif-työkaluun.",
    "api-h2":"API-palvelu",
	"api": "Finto AI toimii myös rajapintapalveluna, joka voidaan integroida omiin järjestelmiin",
	"text-box-label-text":"Kuvailtava teksti",
	"nav-link":"Sisällonkuvailu",	
    "limit":"Ehdotusten määrä",		
    "label-for-project": "Sanasto ja kieli",
	"get-suggestions": "Anna aihe-ehdotukset",
	"suggestions": "Ehdotetut aiheet",
	"footer": "Lomakkeen ja rajapintapalveluiden kautta lähettyjä tietoja ei talleteta.  Palvelun käyttöä seurataan ja tilastoidaan palvelun kehittämiseksi."
}


  } ).done(function() {

    set_locale_to(url('?locale'));

    History.Adapter.bind(window, 'statechange', function(){

      set_locale_to(url('?locale'));

    });

    $('.switch-locale').on('click', 'a', function(e) {

      e.preventDefault();

      History.pushState(null, null, "?locale=" + $(this).data('locale'));

    });

  });

});



/* jQuery(function($) {
  $.i18n().load({
    'en': {	"api-title": "API service",
	"nav-about": "About",
	"nav-feedback": "Feedback",
	"splash": "Finto AI will suggest subjects for a given text. It's based on Annif, a tool for automated subject indexing.",
    "api-h2":"API Service",
	"api": "Finto AI is also an API service that can be integrated to other systems.",
	"text-box-label-text":"Submit text to be indexed below",
	"nav-link":"Subject indexing",	
    "limit":"Number of suggestions",		
    "label-for-project": "Vocabulary and language",
	"get-suggestions": "Get suggestions",
	"suggestions": "Suggestions",
	"footer": "The data submitted via the above form or the API will not be saved anywhere. The use of the service is being monitored for development purposes."

},
	'sv': {	"api-title": "API service",
	"nav-about": "Information",
	"nav-feedback": "Respons",
	"splash": "Finto AI föreslår deskriptorer / indexeringstermer / ämnesord för text. Det är baserat på Annif, som är ett verktyg för automatiskt indexering",
    "api-h2":"API Service",
	"api": "Finto AI är också ett API service som man kan integrera med sina egna system.",
	"text-box-label-text":"Text",
	"nav-link":"Innehållsbeskrivning / ämnesordsindexering",	
    "limit":"Antal föreslag",		
    "label-for-project": "Volabulär och språk",
	"get-suggestions": "Analysera",
	"suggestions": "Föreslag",
	"footer": "Uppgifterna som skickas via ovanstående formuläret eller API kommer inte att sparas eller lagras. Användningen av tjänsten övervakas för utvecklingsändamål."
}
  
  }).done(function() {
    $('#splash').text($.i18n('splash'));
    $('#nav-about').text($.i18n('nav-about'));
	$('#nav-feedback').text($.i18n('nav-feedback'));
	$('#api-title').text($.i18n('api-title'));
	$('#text-box-label-text').text($.i18n('text-box-label-text'));
	$('#nav-link').text($.i18n('nav-link'));
    $('#limit').text($.i18n('limit'));	
	$('#label-for-project').text($.i18n('label-for-project'));
	$('#footer').text($.i18n('footer'));
	$('#api').text($.i18n('api'));
	$('#api-h2').text($.i18n('api-h2'));
	$('#get-suggestions').text($.i18n('get-suggestions'));
	$('#suggestions').text($.i18n('suggestions'));

	// <---
  });
});


   
	
 $.i18n( {

    locale:'sv' 	// Locale is English
} );  */
 