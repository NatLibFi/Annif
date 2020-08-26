
if (window.location.protocol.startsWith('http')) {
    // http or https - use API of current Annif instance
    var base_url = '/v1/'; 
} else {
    // local development case - use Finto AI dev API
    var base_url = 'https://ai.dev.finto.fi/v1/';
}

function clearResults() {
    $('#results').empty();
    $('#suggestions').hide();
}

function fetchProjects() {
    $.ajax({
        url: base_url + "projects",
        method: 'GET',
        success: function(data) {
            $('#project').empty();
            $.each(data.projects, function(idx, value) {
                $('#project').append(
                    $('<option>').attr('value',value.project_id).append(value.name)
                );
            });
        }
    });
}

function makeLabelLanguageOptions() {
    $('#label-language').append(
        $('<option>').attr('value','project-language').attr('data-i18n','label-language-option-project'),
        $('<option>').attr('value','fi').attr('data-i18n','label-language-option-fi'),
        $('<option>').attr('value','sv').attr('data-i18n','label-language-option-sv'),
        $('<option>').attr('value','en').attr('data-i18n','label-language-option-en'),
    );
}

function getLabelPromise(uri, lang) {
    return $.ajax({
        // TODO: Define base url for api.finto somewhere else?
        // TODO: How to handle URI to YSO-places, which is not found from YSO?
        // TODO: Also modify the URI to point to the language version of YSO corresponding to UI language (via data-i18n)?
        url: "http://api.finto.fi/rest/v1/yso/label?uri=" + uri + "&lang=" + lang,
        method: 'GET'
    });
}

function showResults(data) {
    $.each(data.results, function(idx, value) {
        $('#no-results').hide();
        $('#results').append(
            $('<li class="list-group-item p-0">').append(
                $('<meter class="mr-2">').attr('value',value.score).attr('max',1.0).attr('title',value.score.toFixed(4)),
                $('<a target="_blank">').attr('href',value.uri).append(value.label)
            )
        );
        $('#results').show();
    });
}

function getSuggestions() {
    $('#suggestions').show();
    $('#results-spinner').show();
    $.ajax({
        url: base_url + "projects/" + $('#project').val() + "/suggest",
        method: 'POST',
        data: {
          text: $('#text').val(),
          limit: $('input[name="limit"]:checked').val(),
          threshold: 0.01
        },
        success: function(data) {
            $('#results-spinner').hide();

            if (data.results.length == 0) {
                $('#results').hide();
                $('#no-results').show();
            }

            if ($('#label-language').val() == 'project-language') {
                showResults(data);
            }
            else {
                var promises = []
                $.each(data.results, function(idx, value) {
                    promises.push(
                        getLabelPromise(value.uri, $('#label-language').val())
                    );
                });
                $.when.apply($, promises).done(function(result) {
                    $.each(promises, function(idx, promise) {
                        // TODO: Make sure label is taken from the right URI (promises and results are in the same order)
                        // TODO: What to do if api.finto does not respond?
                        data.results[idx].label = promise.responseJSON.prefLabel;
                    });
                    showResults(data);
                });
            }
        }
    });
}

function disableButton() {
    $('#get-suggestions').prop("disabled", true);
}

function enableButton() {
    $('#get-suggestions').prop("disabled", false);
}

$(document).ready(function() {
    $('#no-results').hide();
    $('#results-spinner').hide();
    clearResults();
    if ($.trim($('#text').val()) != "") {
        enableButton();
    } else {
        disableButton();
    }
    fetchProjects();
    makeLabelLanguageOptions();
    $('#get-suggestions').click(function() {
        clearResults();
        getSuggestions();
    });
    $('#button-clear').click(function() {
        $('#text').val('');
        $('#text').focus();
        clearResults();
        disableButton();
    });
    $('#text').on("input", function() {
        enableButton();
    });
});
