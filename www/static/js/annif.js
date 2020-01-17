
var base_url = 'https://api.annif.org/v1/';

function clearResults() {
    $('#results').empty();
}

function fetch_projects() {
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

function analyze() {
    $.ajax({
        url: base_url + "projects/" + $('#project').val() + "/suggest",
        method: 'POST',
        data: { 
          text: $('#text').val(),
        },
        success: function(data) {
            var firstscore = null;
            $.each(data.results, function(idx, value) {
                if (firstscore == null) { firstscore = value.score }
                $('#results').append(
                    $('<li class="list-group-item p-0">').append(
                        $('<meter class="mr-2">').attr('value',value.score).attr('max',firstscore),
                        $('<a>').attr('href',value.uri).append(value.label)
                    )
                );
            });
        }
    });
}

$(document).ready(function() {
    fetch_projects();
    $('#analyze').click(function() {
        clearResults();
        analyze();
    });
});
