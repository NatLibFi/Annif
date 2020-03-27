
var base_url = 'https://api.annif.org/v1/';

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
        },
        success: function(data) {
            console.log(data);
            var firstscore = null;
            $.each(data.results, function(idx, value) {
                if (firstscore == null) { firstscore = value.score }
                $('#results').append(
                    $('<li class="list-group-item p-0">').append(
                        $('<meter class="mr-2">').attr('value',value.score).attr('max',firstscore).attr('title',value.score),
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
