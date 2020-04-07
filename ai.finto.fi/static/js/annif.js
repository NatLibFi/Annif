
var base_url = 'http://ai.dev.finto.fi/v1/';

function clearResults() {
    $('#results').empty();
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

function getSuggestions() {
    $.ajax({
        url: base_url + "projects/" + $('#project').val() + "/suggest",
        method: 'POST',
        data: {
          text: $('#text').val(),
          limit: $('input[name="limit"]:checked').val(),
          threshold: 0.01
        },
        success: function(data) {
            if (data.results.length == 0) {
                $('#results').append(
                    $('<li class="list-group-item p-0">Ei tuloksia</li>')
                );
            }
            $.each(data.results, function(idx, value) {
                $('#results').append(
                    $('<li class="list-group-item p-0">').append(
                        $('<meter class="mr-2">').attr('value',value.score).attr('max',1.0).attr('title',value.score.toFixed(4)),
                        $('<a>').attr('href',value.uri).append(value.label)
                    )
                );
            });
        }
    });
}

$(document).ready(function() {
    fetchProjects();
    $('#get-suggestions').click(function() {
        clearResults();
        getSuggestions();
    });
});
