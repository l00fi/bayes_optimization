# Байесовская оптимизация для настройки моделей машинного обучения
## Ресурсы 
- [Статья на хэндбуке](https://education.yandex.ru/handbook/ml/article/podbor-giperparametrov)
- [Статья на хабре (1)](https://habr.com/ru/companies/otus/articles/754402/)
- [Лекция автора книжки ниже на счёт байесовской оптимизации](https://youtu.be/ImXOdgEgaTM)
- Nguyen Quan - Bayesian Optimization in Action - книжка про байесовскую оптимизацию с примерами на python. [Сюда же рецензия на книгу для более легкой навигации, тоже хабр (2)](https://habr.com/ru/companies/ssp-soft/articles/868100/)
- Свит Дэвид - Тюнинг систем. В этой книжке упоминается байесовская оптимизация, но в более широком смысле
- [Статья на хабре (3) - один из последних разделов как раз посвящен оптимизации гипперпараметров](https://habr.com/ru/articles/853560/)
- [Книжка из Кембриджской типографии по теме](https://books.google.ru/books?hl=ru&lr=&id=MBCrEAAAQBAJ&oi=fnd&pg=PP1&dq=bayesian+optimization&ots=tkMEnf0duE&sig=oyCaMRwHbrcPGdNfX4m3xV_6G9E&redir_esc=y#v=onepage&q=bayesian%20optimization&f=false)

## Заметки
- Сама байесовская оптимизация будет лежать в отдельном файле (bayes_optimization_func) и там же тестироваться.
- Хочу поработать с данными и это тоже сделаю в отдельном файле (data_work)

## Заметки по книге "Свит Дэвид - Тюнинг систем"
*Нужная глава начинается на 194 странице.*
Байесовская оптимизация - оптимизация черного ящика, в том смысле, что мы не знаем принципов работы оптимизируемой системы.