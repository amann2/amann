---
layout: page
title: Books
permalink: /books/
---

{% for post in site.posts %}
{% if post.categories[0] == "books" %}
<a href="{{ post.url }}">{{ post.title }}</a>
{% endif %}
{% endfor %}