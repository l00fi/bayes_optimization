# Байесовская оптимизация для настройки моделей машинного обучения
## Ресурсы 
- [Статья на хэндбуке](https://education.yandex.ru/handbook/ml/article/podbor-giperparametrov)
- [Статья на хабре (1)](https://habr.com/ru/companies/otus/articles/754402/)
- [Лекция автора книжки ниже на счёт байесовской оптимизации](https://youtu.be/ImXOdgEgaTM)
- Nguyen Quan - Bayesian Optimization in Action - книжка про байесовскую оптимизацию с примерами на python. [Сюда же рецензия на книгу для более легкой навигации, тоже хабр (2)](https://habr.com/ru/companies/ssp-soft/articles/868100/)
- Свит Дэвид - Тюнинг систем. В этой книжке упоминается байесовская оптимизация, но в более широком смысле
- [Статья на хабре (3) - один из последних разделов как раз посвящен оптимизации гипперпараметров](https://habr.com/ru/articles/853560/)
- [Книжка из Кембриджской типографии по теме](https://books.google.ru/books?hl=ru&lr=&id=MBCrEAAAQBAJ&oi=fnd&pg=PP1&dq=bayesian+optimization&ots=tkMEnf0duE&sig=oyCaMRwHbrcPGdNfX4m3xV_6G9E&redir_esc=y#v=onepage&q=bayesian%20optimization&f=false)
- Иные источники указаны в самой работе (work.ipynb)

## Заметки
- Сама байесовская оптимизация будет лежать в отдельном файле (bayes_optimization_func) и там же тестироваться.
- Файл bayes_opt_class - это финальная реализация байесовской оптимизации в py файле, для того, чтобы можно было импортировать её как библиотеку и удобно пользоваться, хотя тот же код продублирован в work.ipynb  