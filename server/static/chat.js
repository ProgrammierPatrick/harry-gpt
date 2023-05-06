
function send_chat() {
    // read chat box
    const msg = $('#chat-prompt').val();
    $('#chat-prompt').val('');

    // create chat message elements
    const new_request = $('<div class="request"/>');
    new_request.text(msg);
    $('#messages').append(new_request);
    const new_response = $('<div class="response"/>');
    $('#messages').append(new_response);

    // open SSE stream
    var source = new EventSource('/sse/send_chat');
    source.onmessage = function (event) {
        $('.response:last').append(event.data);
    };

    // send chat message
    var encoded_msg = encodeURIComponent(msg);
    $.ajax({
        type: 'POST',
        url: '/ajax/send_chat',
        data: { msg: encoded_msg },
        error: function (xhr, status, error) {
            console.log(xhr, status, error);
        },
        success: function (response) {

        }
    });
}