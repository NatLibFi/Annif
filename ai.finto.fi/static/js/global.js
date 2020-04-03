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

    $.i18n().load({

        'en': {
            "api-title": "API service",
            "nav-about": "About",
            "nav-feedback": "Feedback",
            "annif": "Read more",
            "splash": "Finto AI suggests subjects for a given text. It's based on Annif, a tool for automated subject indexing.",
            "api-h2": "API Service",
            "api": "Finto AI is also an API service that can be integrated to other systems.",
            "text-box-label-text": "Enter text to be indexed",
            "nav-link": "Subject indexing",
            "limit": "Number of suggestions",
            "label-for-project": "Vocabulary and language",
            "get-suggestions": "Get suggestions",
            "suggestions": "Suggestions",
            "footer": "The data submitted via the above form or the API will not be saved anywhere. Usage of the service is being monitored for development purposes."

        },
        'sv': {
            "api-title": "API service",
            "nav-about": "Information",
            "nav-feedback": "Respons",
            "annif": "Läs mer",
            "splash": "Finto AI föreslår ämnesord för text. Det är baserat på Annif, ett verktyg för automatisk indexering.",
            "api-h2": "API Service",
            "api": "Finto AI är också en API-tjänst som kan integreras med andra system.",
            "text-box-label-text": "Text",
            "nav-link": "Ämnesordsindexering",
            "limit": "Antal förslag",
            "label-for-project": "Volabulär och språk",
            "get-suggestions": "Ge förslag till ämnesord",
            "suggestions": "Förslag",
            "footer": "Uppgifterna som skickas via formuläret eller API-tjänsten sparas inte. Användningen av tjänsten övervakas för utvecklingsändamål."
        },

        'fi': {
            "api-title": "API service",
            "nav-about": "Tietoja",
            "nav-feedback": "Palaute",
            "annif": "Lisätietoa",
            "splash": "Finto AI ehdottaa tekstille sopivia aiheita. Palvelu perustuu Annif-työkaluun.",
            "api-h2": "API-palvelu",
            "api": "Finto AI toimii myös rajapintapalveluna, joka voidaan integroida omiin järjestelmiin",
            "text-box-label-text": "Kuvailtava teksti",
            "nav-link": "Sisällonkuvailu",
            "limit": "Ehdotusten määrä",
            "label-for-project": "Sanasto ja kieli",
            "get-suggestions": "Anna aihe-ehdotukset",
            "suggestions": "Ehdotetut aiheet",
            "footer": "Lomakkeen ja rajapintapalveluiden kautta lähettyjä tietoja ei talleteta.  Palvelun käyttöä seurataan ja tilastoidaan palvelun kehittämiseksi."
        }


    }).done(function() {

        set_locale_to(url('?locale'));

        History.Adapter.bind(window, 'statechange', function() {

            set_locale_to(url('?locale'));

        });

        $('.switch-locale').on('click', 'a', function(e) {

            e.preventDefault();

            History.pushState(null, null, "?locale=" + $(this).data('locale'));

        });

    });

});



/* TODO: Make the missing links and make them work  */
