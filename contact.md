---
layout: page
title: Contact
tagline: Share your thoughts  
---

<form id="formaction" method="POST">
    <p>Name: </p><input type="text" name="name"><br />
    <p>Email: </p><input type="email" name="email"><br />
    <p>Message: </p><input type="message" name="message"><br />
    <input type="text" name="_gotcha" style="display:none" />
    <input type="submit" value="Send">
</form>
<script>
    var contactform =  document.getElementById('formaction');
    contactform.setAttribute('action', '//formspree.io/' + 'corypruce' + '@' + 'gmail' + '.' + 'com');
</script>

[Go to the Home Page]({{ site.url }}{{ site.baseurl }})
