function defaultLanguage() {
    var locale = $.i18n().locale;
    var languages = ["fi", "sv", "en"];
    for (let i = 0; i < languages.length; ++i) {
        if (locale == languages[i])
            return languages[i];
        if (locale.startsWith(languages[i] + "-"))
            return languages[i];
    }
    return languages[0];
}


var setLocale = function(locale) {

    if (locale) {
        $.i18n().locale = locale;
    } else {
        $.i18n().locale = defaultLanguage();
    }

    $('body').i18n();
    
    /* switch link targets */
    $('#nav-feedback').attr("href", $.i18n('url-feedback'));

    $('#switch-locale a').show();
    $('a[data-locale=' + $.i18n().locale + ']').hide();

};

jQuery(function() {

    $.i18n().load({

        'en': {
            "nav-about": "About",
            "nav-feedback": "Feedback",
            "read-more": "Read more...",
            "splash": "Finto AI suggests subjects for a given text. It's based on Annif, a tool for automated subject indexing.",
            "api-title": "API service",
            "api-desc": "Finto AI is also an API service that can be integrated to other systems.",
            "text-box-label-text": "Enter text to be indexed",
            "text-box-placeholder": 'Copy text here and press the button "Get subject suggestions"',
            "nav-subject-indexing": "Subject indexing",
            "project": "Vocabulary and text language",
            "limit": "Maximum # of suggestions",
            "get-suggestions": "Get subject suggestions",
            "suggestions": "Suggestions",
            "no-results": "No results",
            "footer-text": "The data submitted via the above form or the API will not be saved anywhere. Usage of the service is being monitored for development purposes.",
            "footer-link": "See our privacy policy",
            "url-feedback": "http://finto.fi/en/feedback"
        },
        'sv': {
            "nav-about": "Information",
            "nav-feedback": "Respons",
            "read-more": "Läs mer...",
            "splash": "Finto AI föreslår ämnesord för text. Det är baserat på Annif, ett verktyg för automatisk indexering.",
            "api-title": "API-tjänst",
            "api-desc": "Finto AI är också en API-tjänst som kan integreras med andra system.",
            "text-box-label-text": "Text för indexering",
            "text-box-placeholder": 'Kopiera text här och tryck på knappen "Ge förslag till ämnesord"',
            "nav-subject-indexing": "Ämnesordsindexering",
            "project": "Vokabulär och textens språk",
            "limit": "Maximalt antal förslag",
            "get-suggestions": "Ge förslag till ämnesord",
            "suggestions": "Förslag",
            "no-results": "Inga resultat",
            "footer-text": "Uppgifterna som skickas via formuläret eller API-tjänsten sparas inte. Användningen av tjänsten övervakas för utvecklingsändamål.",
            "footer-link": "Läs vår sekretesspolicy",
            "url-feedback": "http://finto.fi/sv/feedback"
        },

        'fi': {
            "nav-about": "Tietoja",
            "nav-feedback": "Palaute",
            "read-more": "Lue lisää...",
            "splash": "Finto AI ehdottaa tekstille sopivia aiheita. Palvelu perustuu Annif-työkaluun.",
            "api-title": "API-palvelu",
            "api-desc": "Finto AI toimii myös rajapintapalveluna, joka voidaan integroida omiin järjestelmiin.",
            "text-box-label-text": "Kuvailtava teksti",
            "text-box-placeholder": 'Kopioi tähän tekstiä ja paina "Anna aihe-ehdotukset"-nappia',
            "nav-subject-indexing": "Sisällönkuvailu",
            "project": "Sanasto ja tekstin kieli",
            "limit": "Ehdotusten enimmäismäärä",
            "get-suggestions": "Anna aihe-ehdotukset",
            "suggestions": "Ehdotetut aiheet",
            "no-results": "Ei tuloksia",
            "footer-text": "Lomakkeen ja rajapintapalveluiden kautta lähettyjä tietoja ei talleteta.  Palvelun käyttöä seurataan ja tilastoidaan palvelun kehittämiseksi.",
            "footer-link": "Lue tietosuojaseloste",
            "url-feedback": "http://finto.fi/fi/feedback"
        }

    }).done(function() {
        setLocale(url('?locale'));

        History.Adapter.bind(window, 'statechange', function() {
            setLocale(url('?locale'));
        });

        $('#switch-locale').on('click', 'a', function(e) {
            e.preventDefault();
            History.pushState(null, null, "?locale=" + $(this).data('locale'));
        });
    });
});



/* TODO: Make the missing links and make them work  */
