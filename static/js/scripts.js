$("form[name=signup_form").submit(function (e) {
  var $form = $(this);
  var $error = $form.find(".error");
  var data = $form.serialize();

  $.ajax({
    url: "/user/signup",
    type: "POST",
    data: data,
    dataType: "json",
    success: function (resp) {
      window.location.href = "/attendance";
    },
    error: function (resp) {
      if (resp && resp.responseJSON && resp.responseJSON.error) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      } else {
        console.log("An error occurred:", resp);
      }
    },
  });

  e.preventDefault();
});

$("form[name=login_form").submit(function (e) {
  var $form = $(this);
  var $error = $form.find(".error");
  var data = $form.serialize();

  $.ajax({
    url: "/user/login",
    type: "POST",
    data: data,
    dataType: "json",
    success: function (resp) {
      window.location.href = "/attendance";
    },
    error: function (resp) {
      if (resp && resp.responseJSON && resp.responseJSON.error) {
        $error.text(resp.responseJSON.error).removeClass("error--hidden");
      } else {
        console.log("An error occurred:", resp);
      }
    },
  });

  e.preventDefault();
});
