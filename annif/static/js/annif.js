function clearResults() {
    $('#results').empty();
}

function fetch_projects() {
    $.ajax({
        url: "/v1/projects",
        method: 'GET',
        success: function(data) {
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
        url: "/v1/projects/" + $('#project').val() + "/analyze",
        method: 'POST',
        data: { 
          text: $('#text').val(),
        },
        success: function(data) {
            $.each(data.results, function(idx, value) {
                $('#results').append(
                    $('<li class="list-group-item p-0">').append(
                        $('<meter class="mr-2">').attr('value',value.score),
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
