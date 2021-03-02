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
    $('html').attr('lang', $.i18n().locale);
    
    /* switch link targets */
    $('#nav-feedback').attr("href", $.i18n('url-feedback'));
    $('#nav-about').attr("href", $.i18n('url-read-more'));
    $('#read-more').attr("href", $.i18n('url-read-more'));
    $('#api-more-info').attr("href", $.i18n('url-api-more-info'));
    $('#accessibility-statement').attr("href", $.i18n('url-accessibility-statement'));

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
            "api-more-info": "More information",
            "api-desc-link": "OpenAPI description",
            "text-box-label-text": "Enter text to be indexed",
            "text-box-placeholder": 'Copy text here and press the button "Get subject suggestions"',
            "nav-subject-indexing": "Subject indexing",
            "project": "Vocabulary and text language",
            "limit": "Maximum # of suggestions",
            "label-language": "Suggestions language",
            "label-language-option-project": "same as text language",
            "label-language-option-fi": "Finnish",
            "label-language-option-sv": "Swedish",
            "label-language-option-en": "English",
            "get-suggestions": "Get subject suggestions",
            "suggestions": "Suggestions",
            "no-results": "No results",
            "button-copy-label": "Copy label to clipboard",
            "button-copy-uri": "Copy URI to clipboard",
            "button-copy-label-and-uri": "Copy label, URI and language code to clipboard",
            "footer-text": "The data submitted via the above form or the API will not be saved anywhere. Usage of the service is being monitored for development purposes. ",
            "accessibility-statement": "See the accessibility statement",
            "powered-by": "Powered by ",
            "url-feedback": "http://finto.fi/en/feedback",
            "url-read-more": "https://www.kiwi.fi/x/DYDbCQ",
            "url-api-more-info": "https://www.kiwi.fi/x/h4A_Cg",
            "url-accessibility-statement": "https://www.kiwi.fi/x/LQBTCw",
        },
        'sv': {
            "nav-about": "Information",
            "nav-feedback": "Respons",
            "read-more": "Läs mer...",
            "splash": "Finto AI föreslår ämnesord för text. Det är baserat på Annif, ett verktyg för automatisk indexering.",
            "api-title": "API-tjänst",
            "api-desc": "Finto AI är också en API-tjänst som kan integreras med andra system.",
            "api-more-info": "Mer information",
            "api-desc-link": "OpenAPI -beskrivning",
            "text-box-label-text": "Text för indexering",
            "text-box-placeholder": 'Kopiera text hit och tryck på knappen "Ge förslag till ämnesord"',
            "nav-subject-indexing": "Innehållsbeskrivning",
            "project": "Vokabulär och textens språk",
            "limit": "Maximalt antal förslag",
            "label-language": "Förslagets språk",
            "label-language-option-project": "samma som textens språk",
            "label-language-option-fi": "finska",
            "label-language-option-sv": "svenska",
            "label-language-option-en": "engelska",
            "get-suggestions": "Ge förslag till ämnesord",
            "suggestions": "Förslag",
            "no-results": "Inga resultat",
            "button-copy-label": "Kopiert term till urklipp",
            "button-copy-uri": "Kopiert term och URI till urklipp",
            "button-copy-label-and-uri": "Kopiert term, URI och språkets kod till urklipp",
            "footer-text": "Uppgifterna som skickas via formuläret eller API-tjänsten sparas inte. Användningen av tjänsten följs upp och statistikförs för utvecklingsändamål. ",
            "accessibility-statement": "Se tillgänglighetsutlåtande",
            "powered-by": "Drivs av ",
            "url-feedback": "http://finto.fi/sv/feedback",
            "url-read-more": "https://www.kiwi.fi/x/FoDbCQ",
            "url-api-more-info": "https://www.kiwi.fi/x/iIA_Cg",
            "url-accessibility-statement": "https://www.kiwi.fi/x/KABTCw",
        },

        'fi': {
            "nav-about": "Tietoja",
            "nav-feedback": "Palaute",
            "read-more": "Lue lisää...",
            "splash": "Finto AI ehdottaa tekstille sopivia aiheita. Palvelu perustuu Annif-työkaluun.",
            "api-title": "API-palvelu",
            "api-desc": "Finto AI toimii myös rajapintapalveluna, joka voidaan integroida omiin järjestelmiin.",
            "api-more-info": "Lisätietoja",
            "api-desc-link": "OpenAPI-kuvaus",
            "text-box-label-text": "Kuvailtava teksti",
            "text-box-placeholder": 'Kopioi tähän tekstiä ja paina "Anna aihe-ehdotukset"-nappia',
            "nav-subject-indexing": "Sisällönkuvailu",
            "project": "Sanasto ja tekstin kieli",
            "limit": "Ehdotusten enimmäismäärä",
            "label-language": "Aihe-ehdotusten kieli",
            "label-language-option-project": "sama kuin tekstin kieli",
            "label-language-option-fi": "suomi",
            "label-language-option-sv": "ruotsi",
            "label-language-option-en": "englanti",
            "get-suggestions": "Anna aihe-ehdotukset",
            "suggestions": "Ehdotetut aiheet",
            "no-results": "Ei tuloksia",
            "button-copy-label": "Kopioi termi leikepöydälle",
            "button-copy-uri": "Kopioi termi ja URI leikepöydälle",
            "button-copy-label-and-uri": "Kopioi termi, URI ja kielikoodi leikepöydälle",
            "footer-text": "Lomakkeen ja rajapintapalveluiden kautta lähettyjä tietoja ei talleteta. Palvelun käyttöä seurataan ja tilastoidaan palvelun kehittämiseksi. ",
            "accessibility-statement": "Tutustu saavutettavuusselosteeseen",
            "powered-by": "Voimanlähteenä ",
            "url-feedback": "http://finto.fi/fi/feedback",
            "url-read-more": "https://www.kiwi.fi/x/-oHbCQ",
            "url-api-more-info": "https://www.kiwi.fi/x/VYA_Cg",
            "url-accessibility-statement": "https://www.kiwi.fi/x/LQBTCw",
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
